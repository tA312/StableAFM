#!/usr/bin/env python3
"""Run AFM StableSR with DDPM/DDIM latent-space DPS guidance."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("STABLESR_ROOT", Path(__file__).resolve().parents[1])
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from ldm.models.diffusion.ddpm import space_timesteps
from ldm.models.diffusion.latent_dps import (
    DPSConfig,
    LatentDPSDDIMSampler,
    LatentDPSDDPMSampler,
)
from ldm.util import instantiate_from_config


SUFFIXES = {".npy", ".png", ".tif", ".tiff", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Same-size AFM SwinIR results.")
    parser.add_argument("--measurement-dir", type=Path, required=True,
                        help="Observed low-resolution AFM images.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-suffix", default="")
    parser.add_argument("--measurement-suffix", default="")
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--dps-scale", type=float, default=0.0)
    parser.add_argument(
        "--gradient-mode",
        choices=("full", "detach-denoiser"),
        default="full",
    )
    parser.add_argument("--measurement-scale", type=int, default=4)
    parser.add_argument("--phase-y", type=int, default=0)
    parser.add_argument("--phase-x", type=int, default=0)
    parser.add_argument(
        "--init-mode", choices=("condition", "noise"), default="condition"
    )
    parser.add_argument("--hard-data-consistency", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def logical_stem(path, suffix):
    stem = path.stem
    return stem[:-len(suffix)] if suffix and stem.endswith(suffix) else stem


def image_map(directory, suffix):
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        stem = logical_stem(path, suffix)
        if stem in result:
            raise RuntimeError(f"Duplicate sample name {stem!r} in {directory}.")
        result[stem] = path
    if not result:
        raise RuntimeError(f"No supported AFM images found in {directory}.")
    return result


def discover_pairs(args):
    inputs = image_map(args.input_dir, args.input_suffix)
    measurements = image_map(args.measurement_dir, args.measurement_suffix)
    if set(inputs) != set(measurements):
        raise RuntimeError(
            "Input/measurement names do not match: "
            f"missing_measurements={sorted(set(inputs) - set(measurements))[:10]}, "
            f"unused_measurements={sorted(set(measurements) - set(inputs))[:10]}."
        )
    names = sorted(inputs)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        names = names[:args.max_samples]
    return [(name, inputs[name], measurements[name]) for name in names]


def read_normalized(path):
    if path.suffix.lower() == ".npy":
        image = np.load(str(path), allow_pickle=False)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"Failed to read {path}.")
    if image.ndim != 2:
        raise ValueError(f"{path} must be a two-dimensional grayscale image.")
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    elif image.dtype in (np.float32, np.float64):
        image = image.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype {image.dtype} in {path}.")
    if (
        not np.isfinite(image).all()
        or float(image.min()) < 0.0
        or float(image.max()) > 1.0
    ):
        raise ValueError(f"{path} is not finite and normalized to [0, 1].")
    return image


def load_model(config, checkpoint_path):
    # The full Lightning checkpoint already contains the trained StableSR
    # weights. Avoid requiring its initialization checkpoint again at inference.
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    config.model.params.ckpt_path = None
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain state_dict.")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    gc.collect()
    model.eval().requires_grad_(False)
    return model


def configure_respaced_ddpm(model, steps):
    original_sqrt_alphas = model.sqrt_alphas_cumprod.detach().clone()
    original_sqrt_one_minus = (
        model.sqrt_one_minus_alphas_cumprod.detach().clone()
    )
    if original_sqrt_alphas.numel() != 1000:
        raise RuntimeError("StableAFM expects the original 1000-step schedule.")
    selected = sorted(space_timesteps(1000, [steps]))
    last_alpha_cumprod = 1.0
    betas = []
    for timestep, alpha_cumprod in enumerate(model.alphas_cumprod):
        if timestep in selected:
            betas.append(1.0 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
    betas = np.asarray([value.detach().cpu().numpy() for value in betas])
    model.register_schedule(given_betas=betas, timesteps=len(betas))
    model.num_timesteps = 1000
    model.ori_timesteps = selected
    return original_sqrt_alphas, original_sqrt_one_minus


def fixed_noise(shape, indices, seed, device):
    samples = []
    for index in indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 100000 + int(index))
        samples.append(torch.randn(
            (1,) + tuple(shape[1:]), generator=generator, dtype=torch.float32
        ))
    return torch.cat(samples).to(device)


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_output(output_dir, stem, image):
    npy_dir = output_dir / "npy"
    tiff_dir = output_dir / "tiff16"
    preview_dir = output_dir / "png8"
    for directory in (npy_dir, tiff_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)
    np.save(npy_dir / f"{stem}.npy", image.astype(np.float32))
    image16 = np.rint(np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)
    image8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(tiff_dir / f"{stem}.tiff"), image16):
        raise OSError(f"Failed to save output for {stem}.")
    if not cv2.imwrite(str(preview_dir / f"{stem}.png"), image8):
        raise OSError(f"Failed to save preview for {stem}.")


def main():
    args = parse_args()
    if args.batch_size <= 0 or not 1 <= args.steps <= 1000:
        raise ValueError("Invalid batch size or sampling step count.")
    if args.sampler == "ddpm" and args.eta:
        raise ValueError("--eta is only available with DDIM.")
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("The config and full StableAFM checkpoint are required.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("StableAFM inference requires a CUDA device.")

    dps_config = DPSConfig(
        guidance_scale=args.dps_scale,
        measurement_scale=args.measurement_scale,
        phase_y=args.phase_y,
        phase_x=args.phase_x,
        gradient_mode=args.gradient_mode,
    )
    dps_config.validate()
    pairs = discover_pairs(args)
    config = OmegaConf.load(str(args.config))
    seed_all(args.seed)
    model = load_model(config, args.checkpoint)

    original_sqrt_alphas = model.sqrt_alphas_cumprod.detach().clone()
    original_sqrt_one_minus = (
        model.sqrt_one_minus_alphas_cumprod.detach().clone()
    )
    if args.sampler == "ddpm":
        original_sqrt_alphas, original_sqrt_one_minus = (
            configure_respaced_ddpm(model, args.steps)
        )
    model = model.to(device)
    sampler = (
        LatentDPSDDPMSampler(model)
        if args.sampler == "ddpm"
        else LatentDPSDDIMSampler(model)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.time()
    for batch_number in range(math.ceil(len(pairs) / args.batch_size)):
        batch_pairs = pairs[
            batch_number * args.batch_size:(batch_number + 1) * args.batch_size
        ]
        input_arrays = [read_normalized(item[1]) for item in batch_pairs]
        measurement_arrays = [read_normalized(item[2]) for item in batch_pairs]
        shape = input_arrays[0].shape
        if any(image.shape != shape for image in input_arrays):
            raise ValueError("All inputs in a batch must have the same shape.")
        expected_measurement_shape = (
            len(range(args.phase_y, shape[0], args.measurement_scale)),
            len(range(args.phase_x, shape[1], args.measurement_scale)),
        )
        if any(image.shape != expected_measurement_shape
               for image in measurement_arrays):
            raise ValueError(
                f"Measurements must have shape {expected_measurement_shape}."
            )

        batch_size = len(batch_pairs)
        indices = list(range(
            batch_number * args.batch_size,
            batch_number * args.batch_size + batch_size,
        ))
        inputs = torch.from_numpy(np.stack(input_arrays)).unsqueeze(1).to(device)
        measurements = (
            torch.from_numpy(np.stack(measurement_arrays)).unsqueeze(1).to(device)
        )
        with torch.no_grad():
            input_rgb = inputs.repeat(1, 3, 1, 1).mul(2.0).sub(1.0)
            seed_all(args.seed + 200000 + indices[0])
            struct_latent = model.get_first_stage_encoding(
                model.encode_first_stage(input_rgb)
            )
            conditioning = model.cond_stage_model([""] * batch_size)
            noise = fixed_noise(struct_latent.shape, indices, args.seed, device)
            if args.init_mode == "noise":
                initial_latent = noise
            else:
                original_t = torch.full(
                    (batch_size,), 999, device=device, dtype=torch.long
                )
                initial_latent = model.q_sample_respace(
                    struct_latent,
                    original_t,
                    original_sqrt_alphas,
                    original_sqrt_one_minus,
                    noise=noise,
                )

        seed_all(args.seed + 300000 + indices[0])
        if args.sampler == "ddpm":
            samples, guidance = sampler.sample_dps(
                steps=args.steps,
                shape=struct_latent.shape,
                conditioning=conditioning,
                struct_cond=struct_latent,
                measurement=measurements,
                config=dps_config,
                x_T=initial_latent,
                use_original_timestep_map=True,
                verbose=not args.quiet,
            )
        else:
            samples, guidance = sampler.sample_dps(
                steps=args.steps,
                shape=struct_latent.shape,
                conditioning=conditioning,
                struct_cond=struct_latent,
                measurement=measurements,
                config=dps_config,
                eta=args.eta,
                x_T=initial_latent,
                verbose=not args.quiet,
            )

        with torch.no_grad():
            decoded = ((model.decode_first_stage(samples) + 1.0) * 0.5).clamp(
                0.0, 1.0
            )
            gray = (
                decoded[:, 0:1] * 0.299
                + decoded[:, 1:2] * 0.587
                + decoded[:, 2:3] * 0.114
            )
            if args.hard_data_consistency:
                gray = gray.clone()
                gray[
                    ...,
                    args.phase_y::args.measurement_scale,
                    args.phase_x::args.measurement_scale,
                ] = measurements
            outputs = gray.clamp(0.0, 1.0).squeeze(1).cpu().numpy()

        for local_index, (name, _, _) in enumerate(batch_pairs):
            save_output(args.output_dir, name, outputs[local_index])
            records.append({
                "name": name,
                "final_measurement_mse": guidance["steps"][-1][
                    "measurement_mse"
                ],
            })
        print(f"Processed {len(records)}/{len(pairs)} images", flush=True)

    summary = {
        "count": len(records),
        "checkpoint": str(args.checkpoint.resolve()),
        "sampler": args.sampler,
        "steps": args.steps,
        "dps_scale": args.dps_scale,
        "gradient_mode": args.gradient_mode,
        "measurement_scale": args.measurement_scale,
        "phase": [args.phase_y, args.phase_x],
        "init_mode": args.init_mode,
        "hard_data_consistency": args.hard_data_consistency,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "records": records,
    }
    with (args.output_dir / "summary.json").open("w") as output_file:
        json.dump(summary, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
