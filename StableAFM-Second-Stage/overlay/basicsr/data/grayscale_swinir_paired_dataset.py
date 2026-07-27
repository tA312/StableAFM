"""Paired grayscale SwinIR/HR data for phase-preserving StableSR training."""

import random
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class GrayscaleSwinIRPairedDataset(data.Dataset):
    """Read float32 SwinIR outputs and same-size uint8/uint16 grayscale HR targets.

    SwinIR was run on ``modcrop(HR)[0::scale, 0::scale]``.  Training crops
    therefore start at coordinates divisible by ``crop_alignment`` and no
    geometric augmentation is applied after SwinIR inference.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.lq_root = Path(opt['dataroot_lq'])
        self.gt_root = Path(opt['dataroot_gt'])
        self.phase = opt.get('phase', 'train')
        self.crop_size = int(opt.get('crop_size', 256))
        self.scale = int(opt.get('scale', 4))
        self.crop_alignment = int(opt.get('crop_alignment', self.scale))
        self.repeat_factor = int(opt.get('repeat_factor', 1))

        if self.phase not in {'train', 'val', 'test'}:
            raise ValueError(f'Unsupported phase: {self.phase}')
        if self.crop_size <= 0:
            raise ValueError('crop_size must be positive.')
        if self.scale <= 0 or self.crop_alignment <= 0:
            raise ValueError('scale and crop_alignment must be positive.')
        if self.repeat_factor <= 0:
            raise ValueError('repeat_factor must be positive.')
        if self.phase != 'train' and self.repeat_factor != 1:
            raise ValueError('repeat_factor may only be used for the training split.')
        if self.crop_size % self.crop_alignment:
            raise ValueError('crop_size must be divisible by crop_alignment.')
        if not self.lq_root.is_dir():
            raise FileNotFoundError(self.lq_root)
        if not self.gt_root.is_dir():
            raise FileNotFoundError(self.gt_root)

        lq_by_stem = {path.stem: path for path in self.lq_root.glob('*.npy')}
        gt_by_stem = {
            path.stem: path
            for path in self.gt_root.iterdir()
            if path.is_file() and path.suffix.lower() in {
                '.npy', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'
            }
        }
        missing_lq = sorted(set(gt_by_stem) - set(lq_by_stem))
        missing_gt = sorted(set(lq_by_stem) - set(gt_by_stem))
        if missing_lq or missing_gt:
            raise RuntimeError(
                'Unpaired grayscale data: '
                f'missing_lq={missing_lq[:10]}, missing_gt={missing_gt[:10]}'
            )

        stems = sorted(lq_by_stem)
        expected_count = opt.get('expected_count')
        if expected_count is not None and len(stems) != int(expected_count):
            raise RuntimeError(
                f'Expected {int(expected_count)} pairs in {self.lq_root}, found {len(stems)}.'
            )
        if not stems:
            raise RuntimeError(f'No paired data found in {self.lq_root}.')

        max_samples = opt.get('max_samples')
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError('max_samples must be positive when provided.')
            # ``stems`` is sorted above, so validation/test subsets are stable
            # and always select the same first N paired images.
            stems = stems[:max_samples]
        self.paths = [(lq_by_stem[stem], gt_by_stem[stem]) for stem in stems]

    def _crop_origin(self, height, width):
        max_top = height - self.crop_size
        max_left = width - self.crop_size
        if max_top < 0 or max_left < 0:
            raise ValueError(
                f'Image size {(height, width)} is smaller than crop size {self.crop_size}.'
            )
        if self.phase == 'train':
            top = random.randint(0, max_top // self.crop_alignment) * self.crop_alignment
            left = random.randint(0, max_left // self.crop_alignment) * self.crop_alignment
        else:
            top = (max_top // 2 // self.crop_alignment) * self.crop_alignment
            left = (max_left // 2 // self.crop_alignment) * self.crop_alignment
        return top, left

    def __getitem__(self, index):
        # Logical repetition increases independently sampled aligned crops
        # without duplicating the full-resolution NPY files on GPFS.
        lq_path, gt_path = self.paths[index % len(self.paths)]
        img_lq = np.load(str(lq_path), mmap_mode='r', allow_pickle=False)
        if img_lq.dtype != np.float32 or img_lq.ndim != 2:
            raise ValueError(
                f'{lq_path} must be a 2D float32 array, got {img_lq.dtype} {img_lq.shape}.'
            )

        if gt_path.suffix.lower() == '.npy':
            img_gt = np.load(str(gt_path), mmap_mode='r', allow_pickle=False)
            if img_gt.dtype not in (np.uint8, np.uint16) or img_gt.ndim != 2:
                raise ValueError(
                    f'{gt_path} must be a 2D uint8/uint16 array, '
                    f'got {img_gt.dtype} {img_gt.shape}.'
                )
            gt_denominator = float(np.iinfo(img_gt.dtype).max)
        else:
            img_gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if img_gt is None:
                raise OSError(f'Failed to read {gt_path}.')
            gt_denominator = 255.0
        gt_height = img_gt.shape[0] - img_gt.shape[0] % self.scale
        gt_width = img_gt.shape[1] - img_gt.shape[1] % self.scale
        img_gt = img_gt[:gt_height, :gt_width]
        if img_lq.shape != img_gt.shape:
            raise ValueError(
                f'Shape mismatch for {lq_path.name}: LQ {img_lq.shape}, modcropped GT {img_gt.shape}.'
            )

        top, left = self._crop_origin(*img_lq.shape)
        bottom = top + self.crop_size
        right = left + self.crop_size
        # Copy the mmap slice so torch receives writable, contiguous memory.
        lq_crop = np.array(img_lq[top:bottom, left:right], dtype=np.float32, copy=True)
        gt_crop = (
            np.array(img_gt[top:bottom, left:right], dtype=np.float32, copy=True)
            / gt_denominator
        )
        if not np.isfinite(lq_crop).all():
            raise ValueError(f'{lq_path} contains non-finite values in the selected crop.')
        if lq_crop.min() < 0.0 or lq_crop.max() > 1.0:
            raise ValueError(f'{lq_path} contains values outside [0, 1].')

        return {
            'lq': torch.from_numpy(lq_crop).unsqueeze(0),
            'gt': torch.from_numpy(gt_crop).unsqueeze(0),
            'lq_path': str(lq_path),
            'gt_path': str(gt_path),
            'crop_top': top,
            'crop_left': left,
        }

    def __len__(self):
        return len(self.paths) * self.repeat_factor


@DATASET_REGISTRY.register()
class AFMMultimodalSwinIRPairedDataset(data.Dataset):
    """Read fixed-size uint16 AFM SwinIR/HR pairs from multiple modalities.

    The on-disk layout is ``root/split/modality/npy/<stem>.npy``.  A sample is
    keyed by both modality and stem, so different modalities may reuse a
    filename.  If ``modalities`` is omitted, modality directories are
    discovered from the low-quality root.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.lq_root = Path(opt['dataroot_lq'])
        self.gt_root = Path(opt['dataroot_gt'])
        self.phase = opt.get('phase', 'train')
        self.crop_size = int(opt.get('crop_size', 256))
        configured_modalities = opt.get('modalities')
        if configured_modalities is None:
            split_root = self.lq_root / self.phase
            self.modalities = tuple(sorted(
                path.name for path in split_root.iterdir() if path.is_dir()
            )) if split_root.is_dir() else ()
        else:
            self.modalities = tuple(str(value) for value in configured_modalities)
        self.expected_counts = {
            str(key): int(value)
            for key, value in opt.get('expected_counts', {}).items()
        }
        per_modality_limit = opt.get('max_samples_per_modality')
        if isinstance(per_modality_limit, Mapping):
            self.max_samples_per_modality = {
                str(key): int(value) for key, value in per_modality_limit.items()
            }
        elif per_modality_limit is None:
            self.max_samples_per_modality = {}
        else:
            self.max_samples_per_modality = {
                modality: int(per_modality_limit) for modality in self.modalities
            }

        if self.phase not in {'train', 'val', 'test'}:
            raise ValueError(f'Unsupported phase: {self.phase}')
        if self.crop_size <= 0:
            raise ValueError('crop_size must be positive.')
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError('modalities must be a non-empty list of unique names.')
        if any(value <= 0 for value in self.max_samples_per_modality.values()):
            raise ValueError('max_samples_per_modality values must be positive.')
        if not self.lq_root.is_dir():
            raise FileNotFoundError(self.lq_root)
        if not self.gt_root.is_dir():
            raise FileNotFoundError(self.gt_root)

        self.paths = []
        self.modality_counts = {}
        self.available_modality_counts = {}
        for modality in self.modalities:
            lq_dir = self.lq_root / self.phase / modality / 'npy'
            gt_dir = self.gt_root / self.phase / modality / 'npy'
            if not lq_dir.is_dir():
                raise FileNotFoundError(lq_dir)
            if not gt_dir.is_dir():
                raise FileNotFoundError(gt_dir)

            lq_by_stem = {path.stem: path for path in lq_dir.glob('*.npy')}
            gt_by_stem = {path.stem: path for path in gt_dir.glob('*.npy')}
            missing_lq = sorted(set(gt_by_stem) - set(lq_by_stem))
            missing_gt = sorted(set(lq_by_stem) - set(gt_by_stem))
            if missing_lq or missing_gt:
                raise RuntimeError(
                    f'Unpaired AFM {self.phase}/{modality} data: '
                    f'missing_lq={missing_lq[:10]}, missing_gt={missing_gt[:10]}'
                )

            stems = sorted(lq_by_stem)
            expected = self.expected_counts.get(modality)
            if expected is not None and len(stems) != expected:
                raise RuntimeError(
                    f'Expected {expected} pairs for {self.phase}/{modality}, '
                    f'found {len(stems)}.'
                )
            if not stems:
                raise RuntimeError(f'No AFM pairs found for {self.phase}/{modality}.')
            self.available_modality_counts[modality] = len(stems)
            limit = self.max_samples_per_modality.get(modality)
            if limit is not None:
                stems = stems[:limit]
            self.modality_counts[modality] = len(stems)
            self.paths.extend(
                (modality, stem, lq_by_stem[stem], gt_by_stem[stem])
                for stem in stems
            )

    @staticmethod
    def _read_uint16(path):
        image = np.load(str(path), mmap_mode='r', allow_pickle=False)
        if image.dtype != np.uint16 or image.ndim != 2:
            raise ValueError(
                f'{path} must be a 2D uint16 array, got {image.dtype} {image.shape}.'
            )
        return image

    def __getitem__(self, index):
        modality, stem, lq_path, gt_path = self.paths[index]
        img_lq = self._read_uint16(lq_path)
        img_gt = self._read_uint16(gt_path)
        expected_shape = (self.crop_size, self.crop_size)
        if img_lq.shape != expected_shape or img_gt.shape != expected_shape:
            raise ValueError(
                f'{modality}/{stem} must be exactly {expected_shape}; '
                f'got LQ {img_lq.shape}, GT {img_gt.shape}.'
            )

        denominator = float(np.iinfo(np.uint16).max)
        lq = np.array(img_lq, dtype=np.float32, copy=True) / denominator
        gt = np.array(img_gt, dtype=np.float32, copy=True) / denominator
        return {
            'lq': torch.from_numpy(lq).unsqueeze(0),
            'gt': torch.from_numpy(gt).unsqueeze(0),
            'modality': modality,
            'sample_id': f'{modality}/{stem}',
            'lq_path': str(lq_path),
            'gt_path': str(gt_path),
            'crop_top': 0,
            'crop_left': 0,
            'dataset_index': index,
        }

    def __len__(self):
        return len(self.paths)
