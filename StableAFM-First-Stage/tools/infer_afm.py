#!/usr/bin/env python3
"""Run AFM-SwinIR x4 inference on one grayscale image, NPY, CSV, or TXT."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--config', type=Path, default=ROOT / 'configs' / 'finetune_afm_x4.json'
    )
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--kair-root',
        type=Path,
        default=ROOT / 'KAIR',
        help='KAIR checkout after applying this project overlay.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='auto, cpu, cuda, or a CUDA device such as cuda:0.',
    )
    parser.add_argument(
        '--hard-project',
        action='store_true',
        help='Force SR[0::scale,0::scale] to equal the measured LR values.',
    )
    return parser.parse_args()


def read_input(path):
    suffix = path.suffix.lower()
    if suffix == '.npy':
        array = np.load(path, allow_pickle=False)
    elif suffix in ('.csv', '.txt'):
        with path.open(encoding='utf-8-sig') as input_file:
            first_line = input_file.readline()
        delimiter = ',' if suffix == '.csv' or ',' in first_line else None
        array = np.genfromtxt(path, delimiter=delimiter)
        if array.ndim == 2 and array.shape[1] and np.isnan(array[:, -1]).all():
            array = array[:, :-1]
    else:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise ValueError('Could not read {}'.format(path))

    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError('Expected a 2D grayscale array, got {}'.format(array.shape))
    if not np.isfinite(array).all():
        raise ValueError('{} contains NaN or infinity.'.format(path))

    if np.issubdtype(array.dtype, np.integer):
        dtype_info = np.iinfo(array.dtype)
        offset = 0.0
        value_range = float(dtype_info.max)
        normalization = 'integer range [0, {}]'.format(dtype_info.max)
    elif np.issubdtype(array.dtype, np.floating):
        offset = float(array.min())
        value_range = float(array.max()) - offset
        if value_range <= 0:
            raise ValueError('Cannot normalize a constant floating-point array.')
        normalization = 'per-array min/max'
    else:
        raise TypeError('Unsupported dtype: {}'.format(array.dtype))

    normalized = ((array.astype(np.float64) - offset) / value_range).astype(np.float32)
    if normalized.min() < 0.0 or normalized.max() > 1.0:
        raise ValueError('Normalized input falls outside [0, 1].')
    return np.ascontiguousarray(normalized), {
        'source_dtype': str(array.dtype),
        'normalization': normalization,
        'offset': offset,
        'value_range': value_range,
        'input_min': float(array.min()),
        'input_max': float(array.max()),
    }


def select_device(name):
    if name == 'auto':
        name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable.')
        torch.cuda.set_device(device)
    return device


def unwrap_state_dict(checkpoint):
    state = torch.load(str(checkpoint), map_location='cpu')
    if isinstance(state, dict):
        for key in ('params_ema', 'params', 'state_dict'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError('Checkpoint does not contain a state dict.')
    if state and all(key.startswith('module.') for key in state):
        state = {key[len('module.'):]: value for key, value in state.items()}
    return state


def load_model(config_path, checkpoint, kair_root, device):
    kair_root = kair_root.expanduser().resolve()
    if not (kair_root / 'models' / 'select_network.py').is_file():
        raise FileNotFoundError(
            '{} is not an installed KAIR checkout.'.format(kair_root)
        )
    sys.path.insert(0, str(kair_root))
    from models.select_network import define_G

    with config_path.open() as config_file:
        options = json.load(config_file)
    options['is_train'] = False
    options['dist'] = False
    model = define_G(options)
    model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    return model.to(device).eval(), options


def infer(model, low_quality, scale, window_size):
    height, width = low_quality.shape[-2:]
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    if pad_height or pad_width:
        mode = (
            'reflect'
            if height > pad_height and width > pad_width
            else 'replicate'
        )
        low_quality = F.pad(
            low_quality, (0, pad_width, 0, pad_height), mode=mode
        )
    estimate = model(low_quality)
    return estimate[..., :height * scale, :width * scale]


def save_outputs(output_dir, normalized_sr, physical_sr, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'sr_normalized.npy', normalized_sr, allow_pickle=False)
    np.save(output_dir / 'sr_physical.npy', physical_sr, allow_pickle=False)
    preview = np.round(np.clip(normalized_sr, 0.0, 1.0) * 255.0).astype(np.uint8)
    tiff = np.round(np.clip(normalized_sr, 0.0, 1.0) * 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(output_dir / 'sr_preview.png'), preview):
        raise OSError('Failed to write PNG preview.')
    if not cv2.imwrite(str(output_dir / 'sr_normalized.tiff'), tiff):
        raise OSError('Failed to write TIFF output.')
    with (output_dir / 'summary.json').open('w') as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write('\n')


def main():
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (args.input, args.config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    normalized_lr, mapping = read_input(args.input)
    device = select_device(args.device)
    model, options = load_model(
        args.config, args.checkpoint, args.kair_root, device
    )
    scale = int(options['scale'])
    window_size = int(options['netG']['window_size'])
    if int(options['n_channels']) != 1 or int(options['netG']['in_chans']) != 1:
        raise ValueError('This inference tool requires a one-channel model.')

    low_quality = (
        torch.from_numpy(normalized_lr).unsqueeze(0).unsqueeze(0).to(device)
    )
    started = time.monotonic()
    with torch.inference_mode():
        estimate = infer(model, low_quality, scale, window_size)
        if args.hard_project:
            estimate[..., 0::scale, 0::scale] = low_quality
        estimate = estimate.squeeze(0).squeeze(0).float().clamp_(0.0, 1.0).cpu()

    normalized_sr = np.ascontiguousarray(estimate.numpy(), dtype=np.float32)
    physical_sr = (
        normalized_sr.astype(np.float64) * mapping['value_range']
        + mapping['offset']
    )
    consistency_mae = float(np.mean(np.abs(
        normalized_sr[0::scale, 0::scale] - normalized_lr
    )))
    summary = {
        'input': str(args.input),
        'config': str(args.config),
        'checkpoint': str(args.checkpoint),
        'device': str(device),
        'scale': scale,
        'input_shape': list(normalized_lr.shape),
        'output_shape': list(normalized_sr.shape),
        **mapping,
        'hard_projection': bool(args.hard_project),
        'decimation_consistency_mae': consistency_mae,
        'output_normalized_min': float(normalized_sr.min()),
        'output_normalized_max': float(normalized_sr.max()),
        'output_physical_min': float(physical_sr.min()),
        'output_physical_max': float(physical_sr.max()),
        'elapsed_seconds': time.monotonic() - started,
    }
    save_outputs(args.output_dir, normalized_sr, physical_sr, summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
