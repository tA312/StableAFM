"""Latent-space diffusion posterior sampling for grayscale StableSR.

The measurement model used here is deliberately exact and phase preserving:

    y = gray((VAE.decode(z_0) + 1) / 2)[..., phase_y::scale, phase_x::scale]

Both samplers implement the paper-style epsilon correction.  If ``L`` is a
positive L2 measurement loss, the correction has a *plus* sign,

    eps_guided = eps + guidance_scale * sqrt(1 - alpha_bar_t) * grad_z_t L,

so that the corresponding clean-latent estimate moves down the loss gradient.
The denoiser and VAE weights remain frozen; gradients are requested only for
the current noisy latent.
"""

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import noise_like


@dataclass(frozen=True)
class DPSConfig:
    guidance_scale: float
    measurement_scale: int = 4
    phase_y: int = 0
    phase_x: int = 0
    gradient_mode: str = "full"

    def validate(self):
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative.")
        if self.measurement_scale <= 0:
            raise ValueError("measurement_scale must be positive.")
        if not 0 <= self.phase_y < self.measurement_scale:
            raise ValueError("phase_y must be in [0, measurement_scale).")
        if not 0 <= self.phase_x < self.measurement_scale:
            raise ValueError("phase_x must be in [0, measurement_scale).")
        if self.gradient_mode not in {"full", "detach-denoiser"}:
            raise ValueError("gradient_mode must be 'full' or 'detach-denoiser'.")


def decoded_rgb_to_gray01(decoded):
    """Map a VAE decoder result from [-1, 1] to one-channel grayscale."""
    if decoded.ndim != 4 or decoded.shape[1] not in {1, 3}:
        raise ValueError(
            f"Expected decoded BCHW tensor with 1 or 3 channels, got {tuple(decoded.shape)}."
        )
    image = (decoded + 1.0) * 0.5
    if image.shape[1] == 1:
        return image
    return (
        image[:, 0:1] * 0.299
        + image[:, 1:2] * 0.587
        + image[:, 2:3] * 0.114
    )


def phase_decimate(image, scale=4, phase_y=0, phase_x=0):
    """Apply exact two-dimensional strided sampling without interpolation."""
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image, got {tuple(image.shape)}.")
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if not 0 <= phase_y < scale or not 0 <= phase_x < scale:
        raise ValueError("phase offsets must be in [0, scale).")
    return image[..., phase_y::scale, phase_x::scale]


def latent_measurement(model, pred_z0, config):
    """Decode a clean-latent estimate and apply the grayscale measurement."""
    decoded = model.differentiable_decode_first_stage(pred_z0)
    gray = decoded_rgb_to_gray01(decoded)
    return phase_decimate(
        gray,
        scale=config.measurement_scale,
        phase_y=config.phase_y,
        phase_x=config.phase_x,
    )


def measurement_loss_and_gradient(model, z_t, pred_z0, measurement, config):
    """Return per-image MSE and its full gradient with respect to ``z_t``."""
    simulated = latent_measurement(model, pred_z0, config)
    if simulated.shape != measurement.shape:
        raise ValueError(
            "Measurement shape mismatch: "
            f"simulated {tuple(simulated.shape)} versus observed {tuple(measurement.shape)}."
        )
    per_image_loss = (simulated - measurement).square().flatten(1).mean(1)
    gradient = torch.autograd.grad(
        per_image_loss.sum(), z_t, create_graph=False, retain_graph=False, only_inputs=True
    )[0]
    if not torch.isfinite(per_image_loss).all():
        raise FloatingPointError("Non-finite LS-DPS measurement loss.")
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Non-finite LS-DPS latent gradient.")
    return per_image_loss.detach(), gradient.detach()


def measurement_loss_without_gradient(model, pred_z0, measurement, config):
    """Measure data fidelity without constructing a VAE backward graph."""
    simulated = latent_measurement(model, pred_z0, config)
    if simulated.shape != measurement.shape:
        raise ValueError(
            "Measurement shape mismatch: "
            f"simulated {tuple(simulated.shape)} versus observed {tuple(measurement.shape)}."
        )
    per_image_loss = (simulated - measurement).square().flatten(1).mean(1)
    if not torch.isfinite(per_image_loss).all():
        raise FloatingPointError("Non-finite LS-DPS measurement loss.")
    return per_image_loss.detach()


def epsilon_dps_correction(z_t, epsilon, alpha_bar_t, gradient, guidance_scale):
    """Correct epsilon using a positive L2-loss gradient and recompute z0."""
    sqrt_alpha = alpha_bar_t.sqrt()
    sqrt_one_minus = (1.0 - alpha_bar_t).clamp_min(0.0).sqrt()
    guided_epsilon = epsilon + guidance_scale * sqrt_one_minus * gradient
    guided_z0 = (z_t - sqrt_one_minus * guided_epsilon) / sqrt_alpha
    return guided_epsilon, guided_z0


def _model_output(model, z_t, timestep, conditioning, struct_features,
                  unconditional_conditioning=None, unconditional_guidance_scale=1.0):
    if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
        return model.apply_model(z_t, timestep, conditioning, struct_features)
    z_in = torch.cat([z_t, z_t])
    t_in = torch.cat([timestep, timestep])
    c_in = torch.cat([unconditional_conditioning, conditioning])
    if isinstance(struct_features, dict):
        struct_in = {
            key: torch.cat([value, value]) for key, value in struct_features.items()
        }
    else:
        struct_in = torch.cat([struct_features, struct_features])
    output_uncond, output_cond = model.apply_model(
        z_in, t_in, c_in, struct_in
    ).chunk(2)
    return output_uncond + unconditional_guidance_scale * (output_cond - output_uncond)


def _as_epsilon(model, z_t, model_output, alpha_bar_t):
    sqrt_alpha = alpha_bar_t.sqrt()
    sqrt_one_minus = (1.0 - alpha_bar_t).clamp_min(0.0).sqrt()
    if model.parameterization == "eps":
        return model_output
    if model.parameterization == "v":
        return sqrt_alpha * model_output + sqrt_one_minus * z_t
    if model.parameterization == "x0":
        return (z_t - sqrt_alpha * model_output) / sqrt_one_minus.clamp_min(1e-12)
    raise NotImplementedError(f"Unsupported parameterization: {model.parameterization}")


def _predict_with_gradient(model, z_t, timestep, conditioning, struct_features,
                           alpha_bar_t, measurement, config,
                           unconditional_conditioning=None,
                           unconditional_guidance_scale=1.0):
    """Predict epsilon/z0 and calculate the requested LS-DPS gradient."""
    if config.gradient_mode == "full":
        with torch.enable_grad():
            z_variable = z_t.detach().requires_grad_(True)
            model_output = _model_output(
                model,
                z_variable,
                timestep,
                conditioning,
                struct_features,
                unconditional_conditioning,
                unconditional_guidance_scale,
            )
            epsilon = _as_epsilon(model, z_variable, model_output, alpha_bar_t)
            pred_z0 = (
                z_variable - (1.0 - alpha_bar_t).clamp_min(0.0).sqrt() * epsilon
            ) / alpha_bar_t.sqrt()
            losses, gradient = measurement_loss_and_gradient(
                model, z_variable, pred_z0, measurement, config
            )
        return epsilon.detach(), losses, gradient

    with torch.no_grad():
        model_output = _model_output(
            model,
            z_t,
            timestep,
            conditioning,
            struct_features,
            unconditional_conditioning,
            unconditional_guidance_scale,
        )
        epsilon = _as_epsilon(model, z_t, model_output, alpha_bar_t).detach()
    with torch.enable_grad():
        z_variable = z_t.detach().requires_grad_(True)
        pred_z0 = (
            z_variable - (1.0 - alpha_bar_t).clamp_min(0.0).sqrt() * epsilon
        ) / alpha_bar_t.sqrt()
        losses, gradient = measurement_loss_and_gradient(
            model, z_variable, pred_z0, measurement, config
        )
    return epsilon, losses, gradient


def _predict_without_guidance(model, z_t, timestep, conditioning,
                              struct_features, alpha_bar_t, measurement,
                              config, unconditional_conditioning=None,
                              unconditional_guidance_scale=1.0):
    """Scale-zero baseline: retain measurements but skip all backward work."""
    with torch.no_grad():
        model_output = _model_output(
            model,
            z_t,
            timestep,
            conditioning,
            struct_features,
            unconditional_conditioning,
            unconditional_guidance_scale,
        )
        epsilon = _as_epsilon(model, z_t, model_output, alpha_bar_t)
        pred_z0 = (
            z_t - (1.0 - alpha_bar_t).clamp_min(0.0).sqrt() * epsilon
        ) / alpha_bar_t.sqrt()
        losses = measurement_loss_without_gradient(
            model, pred_z0, measurement, config
        )
    return epsilon.detach(), losses, torch.zeros_like(z_t)


def _stats(losses, gradient):
    norms = gradient.flatten(1).norm(dim=1)
    return {
        "measurement_mse": float(losses.mean().cpu()),
        "gradient_norm": float(norms.mean().cpu()),
        "gradient_norm_max": float(norms.max().cpu()),
    }


class LatentDPSDDIMSampler(DDIMSampler):
    """StableSR DDIM sampler with differentiable latent-space DPS guidance."""

    @torch.no_grad()
    def sample_dps(self, steps, shape, conditioning, struct_cond, measurement,
                   config, eta=0.0, x_T=None, temperature=1.0,
                   unconditional_conditioning=None,
                   unconditional_guidance_scale=1.0, verbose=True):
        config.validate()
        if len(shape) != 4:
            raise ValueError(f"Expected BCHW latent shape, got {tuple(shape)}.")
        if measurement.shape[0] != shape[0] or measurement.shape[1] != 1:
            raise ValueError("Observed LR must have shape [B, 1, H, W].")
        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        device = self.model.betas.device
        image = torch.randn(shape, device=device) if x_T is None else x_T
        time_range = np.flip(self.ddim_timesteps)
        iterator = tqdm(time_range, desc="LS-DPS DDIM", total=len(time_range)) \
            if verbose else time_range
        history = []
        for iteration, step in enumerate(iterator):
            index = len(time_range) - iteration - 1
            timestep = torch.full(
                (shape[0],), int(step), device=device, dtype=torch.long
            )
            image, pred_z0, step_stats = self.p_sample_dps(
                image,
                timestep,
                index,
                conditioning,
                struct_cond,
                measurement,
                config,
                temperature=temperature,
                unconditional_conditioning=unconditional_conditioning,
                unconditional_guidance_scale=unconditional_guidance_scale,
            )
            step_stats["diffusion_timestep"] = int(step)
            history.append(step_stats)
        return image, {"steps": history, "pred_z0": pred_z0}

    @torch.no_grad()
    def p_sample_dps(self, z_t, timestep, index, conditioning, struct_cond,
                     measurement, config, temperature=1.0,
                     unconditional_conditioning=None,
                     unconditional_guidance_scale=1.0):
        if config.guidance_scale == 0.0:
            # Route the control through StableSR's native DDIM step.  Besides
            # avoiding VAE backpropagation, this preserves the exact operation
            # order of the unguided sampler for a strict scale-zero baseline.
            z_prev, pred_z0 = super().p_sample_ddim_sr_t(
                z_t,
                conditioning,
                struct_cond,
                timestep,
                index=index,
                temperature=temperature,
                unconditional_conditioning=unconditional_conditioning,
                unconditional_guidance_scale=unconditional_guidance_scale,
            )
            with torch.no_grad():
                losses = measurement_loss_without_gradient(
                    self.model, pred_z0, measurement, config
                )
            return (
                z_prev.detach(), pred_z0.detach(),
                _stats(losses, torch.zeros_like(z_t)),
            )

        batch = z_t.shape[0]
        device, dtype = z_t.device, z_t.dtype
        # The legacy DDIM helper keeps some schedule arrays as NumPy values.
        # Normalize both NumPy scalars and tensors at this boundary.
        alpha_t = torch.as_tensor(
            self.ddim_alphas[index], device=device, dtype=dtype
        ).view(1, 1, 1, 1)
        alpha_prev = torch.as_tensor(
            self.ddim_alphas_prev[index], device=device, dtype=dtype
        ).view(1, 1, 1, 1)
        sigma_t = torch.as_tensor(
            self.ddim_sigmas[index], device=device, dtype=dtype
        ).view(1, 1, 1, 1)
        alpha_t = alpha_t.expand(batch, 1, 1, 1)
        alpha_prev = alpha_prev.expand(batch, 1, 1, 1)
        sigma_t = sigma_t.expand(batch, 1, 1, 1)

        with torch.no_grad():
            struct_features = self.model.structcond_stage_model(struct_cond, timestep)

        epsilon, losses, gradient = _predict_with_gradient(
            self.model, z_t, timestep, conditioning, struct_features,
            alpha_t, measurement, config, unconditional_conditioning,
            unconditional_guidance_scale,
        )
        guided_epsilon, guided_z0 = epsilon_dps_correction(
            z_t, epsilon, alpha_t, gradient, config.guidance_scale
        )
        direction = (1.0 - alpha_prev - sigma_t.square()).clamp_min(0.0).sqrt() \
            * guided_epsilon
        noise = sigma_t * noise_like(z_t.shape, device, False) * temperature
        z_prev = alpha_prev.sqrt() * guided_z0 + direction + noise
        return z_prev.detach(), guided_z0.detach(), _stats(losses, gradient)


class LatentDPSDDPMSampler:
    """Respaced StableSR DDPM sampler with latent-space DPS guidance."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def sample_dps(self, steps, shape, conditioning, struct_cond, measurement,
                   config, x_T=None, temperature=1.0,
                   use_original_timestep_map=True,
                   unconditional_conditioning=None,
                   unconditional_guidance_scale=1.0, verbose=True):
        config.validate()
        if len(shape) != 4:
            raise ValueError(f"Expected BCHW latent shape, got {tuple(shape)}.")
        if steps > len(self.model.betas):
            raise ValueError(
                f"Requested {steps} DDPM steps but schedule has {len(self.model.betas)}."
            )
        if measurement.shape[0] != shape[0] or measurement.shape[1] != 1:
            raise ValueError("Observed LR must have shape [B, 1, H, W].")
        if use_original_timestep_map:
            if not hasattr(self.model, "ori_timesteps") or len(self.model.ori_timesteps) < steps:
                raise RuntimeError("Respaced DDPM requires model.ori_timesteps.")

        device = self.model.betas.device
        image = torch.randn(shape, device=device) if x_T is None else x_T
        time_range = list(reversed(range(steps)))
        iterator = tqdm(time_range, desc="LS-DPS DDPM", total=steps) \
            if verbose else time_range
        history = []
        pred_z0 = image
        for step in iterator:
            diffusion_timestep = torch.full(
                (shape[0],), step, device=device, dtype=torch.long
            )
            if use_original_timestep_map:
                original_step = int(self.model.ori_timesteps[step])
                model_timestep = torch.full_like(diffusion_timestep, original_step)
            else:
                original_step = step
                model_timestep = diffusion_timestep
            with torch.no_grad():
                struct_features = self.model.structcond_stage_model(
                    struct_cond, model_timestep
                )
            image, pred_z0, step_stats = self.p_sample_dps(
                image,
                diffusion_timestep,
                model_timestep,
                conditioning,
                struct_features,
                measurement,
                config,
                temperature=temperature,
                unconditional_conditioning=unconditional_conditioning,
                unconditional_guidance_scale=unconditional_guidance_scale,
            )
            step_stats["diffusion_timestep"] = step
            step_stats["model_timestep"] = original_step
            history.append(step_stats)
        return image, {"steps": history, "pred_z0": pred_z0}

    @torch.no_grad()
    def p_sample_dps(self, z_t, diffusion_timestep, model_timestep,
                     conditioning, struct_features, measurement, config,
                     temperature=1.0, unconditional_conditioning=None,
                     unconditional_guidance_scale=1.0):
        if config.guidance_scale == 0.0:
            # Use the native StableSR DDPM step so scale zero is bitwise
            # equivalent to the established sampler under identical RNG state.
            z_prev, pred_z0 = self.model.p_sample(
                z_t,
                conditioning,
                struct_features,
                diffusion_timestep,
                clip_denoised=self.model.clip_denoised,
                return_x0=True,
                temperature=temperature,
                t_replace=model_timestep,
                unconditional_conditioning=unconditional_conditioning,
                unconditional_guidance_scale=unconditional_guidance_scale,
            )
            with torch.no_grad():
                losses = measurement_loss_without_gradient(
                    self.model, pred_z0, measurement, config
                )
            return (
                z_prev.detach(), pred_z0.detach(),
                _stats(losses, torch.zeros_like(z_t)),
            )

        alpha_t = self.model.alphas_cumprod[diffusion_timestep] \
            .to(device=z_t.device, dtype=z_t.dtype).view(-1, 1, 1, 1)
        epsilon, losses, gradient = _predict_with_gradient(
            self.model, z_t, model_timestep, conditioning, struct_features,
            alpha_t, measurement, config, unconditional_conditioning,
            unconditional_guidance_scale,
        )
        guided_epsilon, guided_z0 = epsilon_dps_correction(
            z_t, epsilon, alpha_t, gradient, config.guidance_scale
        )
        if self.model.clip_denoised:
            guided_z0 = guided_z0.clamp(-1.0, 1.0)
        model_mean, _, model_log_variance = self.model.q_posterior(
            x_start=guided_z0, x_t=z_t, t=diffusion_timestep
        )
        noise = noise_like(z_t.shape, z_t.device, False) * temperature
        nonzero_mask = (1 - (diffusion_timestep == 0).float()).reshape(
            z_t.shape[0], *((1,) * (z_t.ndim - 1))
        )
        z_prev = model_mean + nonzero_mask \
            * (0.5 * model_log_variance).exp() * noise
        return z_prev.detach(), guided_z0.detach(), _stats(losses, gradient)
