#!/usr/bin/env python3
"""Build leakage-free train/val/test manifests for grayscale SR images."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / 'datasets'
DEFAULT_OUTPUT = ROOT / 'manifests'
SPLITS = ('train', 'val', 'test')
IMAGE_SUFFIXES = {'.png', '.tif', '.tiff'}
FIELDS = ('sample_id', 'modality', 'path', 'sha256', 'preview')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--verify-npy', action='store_true')
    parser.add_argument('--preview-count', type=int, default=4)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Could not read image: {}'.format(path))
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError(
            'Expected a uint8/uint16 grayscale image, got {} {}: {}'.format(
                image.dtype, image.shape, path
            )
        )
    return image


def texture_score(path):
    image = read_image(path)
    data_range = float(np.iinfo(image.dtype).max)
    image = image.astype(np.float32) / data_range
    gradient_x = np.abs(image[:, 1:] - image[:, :-1]).mean()
    gradient_y = np.abs(image[1:, :] - image[:-1, :]).mean()
    return float(gradient_x + gradient_y)


def discover_images(split_root):
    return sorted(
        path for path in split_root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def infer_modality(path, split_root):
    relative_parts = path.relative_to(split_root).parts[:-1]
    named_parts = [part for part in relative_parts if part.lower() != 'images']
    return named_parts[0] if named_parts else 'default'


def make_sample_id(path, split_root):
    relative = path.relative_to(split_root).with_suffix('')
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', '_'.join(relative.parts))


def paired_npy_path(image_path):
    if image_path.parent.name.lower() == 'images':
        return image_path.parent.parent / 'npy' / (image_path.stem + '.npy')
    return image_path.with_suffix('.npy')


def collect_records(dataset_root, verify_npy):
    records = {split: [] for split in SPLITS}
    for split in SPLITS:
        split_root = dataset_root / split
        paths = discover_images(split_root)
        if not paths:
            raise FileNotFoundError('No images found in {}'.format(split_root))
        for path in paths:
            image = read_image(path)
            if verify_npy:
                npy_path = paired_npy_path(path)
                if not npy_path.is_file():
                    raise FileNotFoundError(npy_path)
                array = np.load(npy_path, mmap_mode='r', allow_pickle=False)
                if image.dtype != array.dtype or image.shape != array.shape:
                    raise ValueError('Image/NPY metadata mismatch: {}'.format(path))
                if not np.array_equal(image, array):
                    raise ValueError('Image/NPY values differ: {}'.format(path))
            records[split].append({
                'sample_id': make_sample_id(path, split_root),
                'modality': infer_modality(path, split_root),
                'path': str(path.resolve()),
                'sha256': file_sha256(path),
                'preview': 0,
            })
    return records


def deduplicate(records):
    kept = {}
    reserved_hashes = set()
    for split in ('test', 'val', 'train'):
        kept[split] = []
        for record in records[split]:
            if record['sha256'] in reserved_hashes:
                continue
            reserved_hashes.add(record['sha256'])
            kept[split].append(record)
    return kept


def select_previews(records, count):
    modalities = sorted({record['modality'] for record in records})
    for modality in modalities:
        candidates = [record for record in records if record['modality'] == modality]
        ranked = sorted(
            candidates,
            key=lambda record: (-texture_score(Path(record['path'])), record['sample_id']),
        )
        for record in ranked[:count]:
            record['preview'] = 1


def write_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def count_by_modality(records):
    return dict(sorted(Counter(record['modality'] for record in records).items()))


def assert_disjoint(kept):
    hashes = {
        split: {record['sha256'] for record in rows}
        for split, rows in kept.items()
    }
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = hashes[left] & hashes[right]
        if overlap:
            raise RuntimeError('{} and {} share {} files.'.format(
                left, right, len(overlap)
            ))


def main():
    args = parse_args()
    if args.preview_count < 0:
        raise ValueError('--preview-count must be non-negative.')
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    records = collect_records(dataset_root, args.verify_npy)
    kept = deduplicate(records)
    select_previews(kept['val'], args.preview_count)
    assert_disjoint(kept)

    for split in SPLITS:
        write_manifest(output_dir / '{}.csv'.format(split), kept[split])

    summary = {
        'dataset_root': str(dataset_root),
        'source_counts': {
            split: count_by_modality(records[split]) for split in SPLITS
        },
        'kept_counts': {
            split: count_by_modality(kept[split]) for split in SPLITS
        },
        'removed_duplicates': {
            split: len(records[split]) - len(kept[split]) for split in SPLITS
        },
        'verified_npy_values': bool(args.verify_npy),
    }
    with (output_dir / 'summary.json').open('w') as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write('\n')

    print('Wrote manifests to {}'.format(output_dir))
    print(json.dumps(summary['kept_counts'], indent=2))


if __name__ == '__main__':
    main()
