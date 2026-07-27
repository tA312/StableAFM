import unittest

import torch

from ldm.models.diffusion.latent_dps import (
    DPSConfig,
    LatentDPSDDIMSampler,
    LatentDPSDDPMSampler,
    decoded_rgb_to_gray01,
    epsilon_dps_correction,
    latent_measurement,
    measurement_loss_and_gradient,
    phase_decimate,
)


class _MockModel:
    parameterization = "eps"
    clip_denoised = False

    def __init__(self):
        self.betas = torch.tensor([0.5])
        self.alphas_cumprod = torch.tensor([0.5])
        self.ori_timesteps = [0]

    def structcond_stage_model(self, struct_cond, timestep):
        return struct_cond

    def apply_model(self, z_t, timestep, conditioning, struct_features):
        # Keep a graph connection to z_t while predicting zero noise.
        return z_t * 0.0

    def differentiable_decode_first_stage(self, z0):
        # decoded_rgb_to_gray01 maps this result exactly back to z0.
        return (2.0 * z0 - 1.0).repeat(1, 3, 1, 1)

    def q_posterior(self, x_start, x_t, t):
        zeros = torch.zeros_like(x_start)
        return x_start, zeros, torch.full_like(x_start, -20.0)

    def p_sample(self, x, c, struct_cond, t, **kwargs):
        # Native scale-zero baseline used by LatentDPSDDPMSampler.
        pred_z0 = torch.zeros_like(x)
        if kwargs.get("return_x0"):
            return pred_z0, pred_z0
        return pred_z0


class LatentDPSTest(unittest.TestCase):
    def test_phase_decimation_uses_both_spatial_axes(self):
        coordinates = torch.arange(8 * 12).reshape(1, 1, 8, 12)
        sampled = phase_decimate(coordinates, scale=4, phase_y=0, phase_x=0)
        self.assertTrue(torch.equal(sampled, coordinates[..., 0::4, 0::4]))
        self.assertEqual(sampled.shape, (1, 1, 2, 3))

    def test_rgb_grayscale_mapping_matches_inference_weights(self):
        decoded = torch.zeros(1, 3, 2, 2)
        decoded[:, 0] = 1.0
        gray = decoded_rgb_to_gray01(decoded)
        expected = 0.299 * 1.0 + (0.587 + 0.114) * 0.5
        self.assertTrue(torch.allclose(gray, torch.full_like(gray, expected)))

    def test_positive_loss_gradient_sign_reduces_measurement_error(self):
        model = _MockModel()
        config = DPSConfig(guidance_scale=1.0, measurement_scale=2)
        z_t = torch.zeros(1, 1, 8, 8, requires_grad=True)
        pred_z0 = z_t
        observation = torch.ones(1, 1, 4, 4)
        losses, gradient = measurement_loss_and_gradient(
            model, z_t, pred_z0, observation, config
        )
        epsilon = torch.zeros_like(z_t)
        alpha = torch.full((1, 1, 1, 1), 0.5)
        _, guided_z0 = epsilon_dps_correction(
            z_t.detach(), epsilon, alpha, gradient, config.guidance_scale
        )
        guided_measurement = latent_measurement(model, guided_z0, config)
        guided_loss = (guided_measurement - observation).square().mean()
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertLess(float(guided_loss), float(losses.mean()))

    def test_one_step_ddpm_and_ddim_dps_are_finite_and_guided(self):
        model = _MockModel()
        config = DPSConfig(guidance_scale=1.0, measurement_scale=2)
        shape = (1, 1, 8, 8)
        z_t = torch.zeros(shape)
        observation = torch.ones(1, 1, 4, 4)
        conditioning = torch.zeros(1, 1)
        struct_cond = torch.zeros(shape)

        ddpm = LatentDPSDDPMSampler(model)
        ddpm_result, ddpm_info = ddpm.sample_dps(
            steps=1,
            shape=shape,
            conditioning=conditioning,
            struct_cond=struct_cond,
            measurement=observation,
            config=config,
            x_T=z_t,
            verbose=False,
        )

        ddim = LatentDPSDDIMSampler.__new__(LatentDPSDDIMSampler)
        ddim.model = model
        ddim.ddim_alphas = torch.tensor([0.5])
        ddim.ddim_alphas_prev = torch.tensor([1.0])
        ddim.ddim_sigmas = torch.tensor([0.0])
        ddim_result, _, ddim_stats = ddim.p_sample_dps(
            z_t=z_t,
            timestep=torch.zeros(1, dtype=torch.long),
            index=0,
            conditioning=conditioning,
            struct_cond=struct_cond,
            measurement=observation,
            config=config,
        )

        self.assertTrue(torch.isfinite(ddpm_result).all())
        self.assertTrue(torch.isfinite(ddim_result).all())
        self.assertGreater(float(ddpm_result.mean()), 0.0)
        self.assertGreater(float(ddim_result.mean()), 0.0)
        self.assertEqual(len(ddpm_info["steps"]), 1)
        self.assertEqual(ddim_stats["measurement_mse"], 1.0)

    def test_scale_zero_skips_guidance_and_preserves_baseline_step(self):
        model = _MockModel()
        config = DPSConfig(guidance_scale=0.0, measurement_scale=2)
        shape = (1, 1, 8, 8)
        z_t = torch.zeros(shape)
        observation = torch.ones(1, 1, 4, 4)
        sampler = LatentDPSDDPMSampler(model)
        result, info = sampler.sample_dps(
            steps=1,
            shape=shape,
            conditioning=torch.zeros(1, 1),
            struct_cond=torch.zeros(shape),
            measurement=observation,
            config=config,
            x_T=z_t,
            verbose=False,
        )
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        self.assertEqual(info["steps"][0]["gradient_norm"], 0.0)
        self.assertEqual(info["steps"][0]["measurement_mse"], 1.0)


if __name__ == "__main__":
    unittest.main()
