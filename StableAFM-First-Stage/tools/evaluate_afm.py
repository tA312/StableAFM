#!/usr/bin/env python3
"""Evaluate a full SwinIR checkpoint on leakage-free AFM manifests."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
KAIR_ROOT = Path(
    os.environ.get('AFM_SWINIR_KAIR', str(ROOT / 'KAIR'))
).expanduser().resolve()
sys.path.insert(0, str(KAIR_ROOT))

from data.dataset_sr_decimation import DatasetSRDecimation
from models.select_network import define_G
from utils import utils_image as util
from utils.utils_sr_metrics import (
    calculate_decimation_artifact_metrics,
    hard_project_decimation,
)


DEFAULT_CONFIG = ROOT / 'configs' / 'finetune_afm_x4.json'
METRIC_FIELDS = (
    'psnr', 'ssim', 'mae', 'dc_mae', 'grid4', 'grid8', 'grid16', 'grid32',
    'phase_contrast_mae', 'phase_ratio', 'target_phase_ratio', 'phase_ratio_error',
    'projected_psnr', 'projected_ssim', 'projected_mae', 'projected_dc_mae',
    'projected_grid4', 'projected_grid8', 'projected_grid16', 'projected_grid32',
    'projected_phase_contrast_mae', 'projected_phase_ratio',
    'projected_phase_ratio_error',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--save-all', action='store_true')
    return parser.parse_args()


def average(rows):
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in METRIC_FIELDS
    }


def tensor_to_uint(image, bit_depth):
    return util.tensor2uint16(image) if bit_depth == 16 else util.tensor2uint(image)


def resolve_kair_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (KAIR_ROOT / path).resolve()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for full SwinIR AFM evaluation.')

    with args.config.open() as config_file:
        options = json.load(config_file)
    options['is_train'] = False
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else resolve_kair_path(options['path']['pretrained_netE'])
    )
    manifest = resolve_kair_path(
        options['evaluation']['{}_manifest'.format(args.split)]
    )
    output_dir = args.output or (
        ROOT / 'results' / 'afm_{}_{}'.format(checkpoint.stem, args.split)
    )
    output_dir = output_dir.expanduser().resolve()
    images_dir = output_dir / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.set_device(device)
    model = define_G(options)
    model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'), strict=True)
    model = model.to(device).eval()

    dataset_options = dict(options['datasets']['test'])
    dataset_options.update({
        'phase': 'test',
        'scale': options['scale'],
        'n_channels': options['n_channels'],
        'image_manifest': str(manifest),
        'max_images': 0,
    })
    dataset = DatasetSRDecimation(dataset_options)
    rows = []
    scale = int(options['scale'])

    print('checkpoint={}'.format(checkpoint), flush=True)
    print('manifest={}'.format(manifest), flush=True)
    print('device={}'.format(device), flush=True)

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            low_quality = sample['L'].unsqueeze(0).to(device)
            target = sample['H']
            estimate = model(low_quality).squeeze(0).float().cpu()
            projected = hard_project_decimation(estimate, sample['L'], scale)
            bit_depth = int(sample['bit_depth'])
            data_range = 65535.0 if bit_depth == 16 else 255.0
            estimate_img = tensor_to_uint(estimate, bit_depth)
            projected_img = tensor_to_uint(projected, bit_depth)
            target_img = tensor_to_uint(target, bit_depth)
            artifact = calculate_decimation_artifact_metrics(
                estimate, target, sample['L'], scale
            )
            projected_artifact = calculate_decimation_artifact_metrics(
                projected, target, sample['L'], scale
            )

            row = {
                'sample_id': sample['sample_id'],
                'modality': sample['group'],
                'path': sample['H_path'],
                'psnr': util.calculate_psnr(
                    estimate_img, target_img, border=scale, data_range=data_range
                ),
                'ssim': util.calculate_ssim(
                    estimate_img, target_img, border=scale, data_range=data_range
                ),
                'mae': torch.mean(torch.abs(estimate - target)).item(),
                'dc_mae': artifact['dc_mae'],
                'grid4': artifact['grid_excess'][4],
                'grid8': artifact['grid_excess'][8],
                'grid16': artifact['grid_excess'][16],
                'grid32': artifact['grid_excess'][32],
                'phase_contrast_mae': artifact['phase_contrast_mae'],
                'phase_ratio': artifact['phase_ratio'],
                'target_phase_ratio': artifact['target_phase_ratio'],
                'phase_ratio_error': artifact['phase_ratio_error'],
                'projected_psnr': util.calculate_psnr(
                    projected_img, target_img, border=scale, data_range=data_range
                ),
                'projected_ssim': util.calculate_ssim(
                    projected_img, target_img, border=scale, data_range=data_range
                ),
                'projected_mae': torch.mean(torch.abs(projected - target)).item(),
                'projected_dc_mae': projected_artifact['dc_mae'],
                'projected_grid4': projected_artifact['grid_excess'][4],
                'projected_grid8': projected_artifact['grid_excess'][8],
                'projected_grid16': projected_artifact['grid_excess'][16],
                'projected_grid32': projected_artifact['grid_excess'][32],
                'projected_phase_contrast_mae': projected_artifact['phase_contrast_mae'],
                'projected_phase_ratio': projected_artifact['phase_ratio'],
                'projected_phase_ratio_error': projected_artifact['phase_ratio_error'],
            }
            rows.append(row)
            if args.save_all or sample['save_preview']:
                util.imsave(
                    estimate_img,
                    str(images_dir / '{}_SR.png'.format(sample['sample_id'])),
                )
            print(
                '[{:03d}/{:03d}] {} [{}] PSNR={:.4f}'.format(
                    index + 1, len(dataset), sample['sample_id'],
                    sample['group'], row['psnr']
                ),
                flush=True,
            )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row['modality']].append(row)
    summary = {
        'checkpoint': str(checkpoint),
        'config': str(args.config.resolve()),
        'split': args.split,
        'manifest': str(manifest),
        'count': len(rows),
        'weighted': average(rows),
        'modalities': {
            group: {'count': len(group_rows), **average(group_rows)}
            for group, group_rows in sorted(grouped.items())
        },
    }

    metrics_path = output_dir / 'per_image_metrics.csv'
    with metrics_path.open('w', newline='') as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'summary.json').open('w') as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write('\n')

    print('output={}'.format(output_dir), flush=True)
    print(
        'WEIGHTED PSNR={:.6f} SSIM={:.6f} MAE={:.6e}'.format(
            summary['weighted']['psnr'], summary['weighted']['ssim'],
            summary['weighted']['mae']
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
