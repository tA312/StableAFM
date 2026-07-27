import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch.utils.data as data

import utils.utils_image as util


class DatasetSRDecimation(data.Dataset):
    """Grayscale x4 SR dataset with on-the-fly decimation degradation.

    With ``preserve_decimation_phase`` enabled, augmentation is applied to HR
    before LR generation so that LR always equals degraded_HR[0::scale, 0::scale].
    The legacy post-degradation paired augmentation remains the default for
    compatibility with existing experiments.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_channels = opt.get('n_channels') or 1
        self.sf = opt.get('scale') or 4
        self.patch_size = opt.get('H_size') or 256
        self.blur_prob = opt.get('blur_prob', 0.1)
        self.blur_sigma = opt.get('blur_sigma', [0.5, 1.5])
        self.preserve_decimation_phase = bool(opt.get('preserve_decimation_phase', False))
        self.lr_noise_std = float(opt.get('lr_noise_std') or 0.0)
        self.lr_mask_prob = float(opt.get('lr_mask_prob') or 0.0)
        self.lr_mask_noise_std = float(opt.get('lr_mask_noise_std') or 0.0)
        if not 0.0 <= self.lr_mask_prob <= 1.0:
            raise ValueError('lr_mask_prob must be between 0 and 1.')
        self.samples = self._load_samples(opt)
        assert self.samples, 'Error: H path is empty.'
        max_images = int(opt.get('max_images') or 0)
        include_image_stems = set(opt.get('include_image_stems') or [])
        if max_images > 0 and include_image_stems:
            included = [
                sample for sample in self.samples
                if Path(sample['path']).stem in include_image_stems
            ]
            assert len(included) == len(include_image_stems), (
                'Could not find all include_image_stems: {}'.format(sorted(include_image_stems))
            )
            included_paths = {sample['path'] for sample in included}
            regular = [sample for sample in self.samples if sample['path'] not in included_paths]
            self.samples = regular[:max(0, max_images - len(included))] + included[:max_images]
        elif max_images > 0:
            self.samples = self.samples[:max_images]
        self.paths_H = [sample['path'] for sample in self.samples]

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ('1', 'true', 'yes', 'y')

    def _load_samples(self, opt):
        manifest_path = opt.get('image_manifest')
        if manifest_path:
            manifest_path = Path(manifest_path).expanduser().resolve()
            samples = []
            with manifest_path.open(newline='', encoding='utf-8-sig') as manifest_file:
                reader = csv.DictReader(manifest_file)
                required = {'path', 'modality'}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError(
                        'Manifest {} must contain columns: {}'.format(
                            manifest_path, ', '.join(sorted(required))
                        )
                    )
                for row in reader:
                    path = Path(row['path']).expanduser()
                    if not path.is_absolute():
                        path = manifest_path.parent / path
                    path = path.resolve()
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    modality = row['modality'].strip() or 'default'
                    sample_id = row.get('sample_id', '').strip()
                    if not sample_id:
                        sample_id = '{}_{}'.format(modality, path.stem)
                    samples.append({
                        'path': str(path),
                        'group': modality,
                        'sample_id': sample_id,
                        'save_preview': self._as_bool(row.get('preview', False)),
                    })
            return samples

        dataroots = opt.get('dataroot_H')
        if isinstance(dataroots, str):
            dataroots = [dataroots]
        if not dataroots:
            return []
        labels = opt.get('dataroot_H_labels') or ['default'] * len(dataroots)
        if len(labels) != len(dataroots):
            raise ValueError('dataroot_H_labels must match dataroot_H length.')
        samples = []
        for dataroot, label in zip(dataroots, labels):
            for path in util.get_image_paths(dataroot):
                samples.append({
                    'path': path,
                    'group': str(label),
                    'sample_id': '{}_{}'.format(label, Path(path).stem),
                    'save_preview': False,
                })
        return samples

    def _pad_if_needed(self, img):
        h, w = img.shape[:2]
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
        return img

    def _crop_train_hr(self, img):
        img = self._pad_if_needed(img)
        h, w = img.shape[:2]
        rnd_h = random.randint(0, h - self.patch_size)
        rnd_w = random.randint(0, w - self.patch_size)
        return img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

    def _modcrop(self, img):
        h, w = img.shape[:2]
        h = h - h % self.sf
        w = w - w % self.sf
        return img[:h, :w, :]

    def _degrade(self, img_H):
        img_for_lq = img_H
        if self.opt['phase'] == 'train' and self.blur_prob > 0 and random.random() < self.blur_prob:
            sigma = random.uniform(float(self.blur_sigma[0]), float(self.blur_sigma[1]))
            blurred = cv2.GaussianBlur(np.squeeze(img_H), (0, 0), sigmaX=sigma, sigmaY=sigma)
            img_for_lq = np.expand_dims(blurred, axis=2)
        return img_for_lq[0::self.sf, 0::self.sf, :]

    def __getitem__(self, index):
        sample = self.samples[index]
        H_path = sample['path']
        img_H = util.imread_uint(H_path, self.n_channels)
        bit_depth = 16 if img_H.dtype == np.uint16 else 8

        if self.opt['phase'] == 'train':
            img_H = self._crop_train_hr(img_H)
            mode = random.randint(0, 7)
            if self.preserve_decimation_phase:
                img_H = util.augment_img(img_H, mode=mode)
                img_L = self._degrade(img_H)
            else:
                img_L = self._degrade(img_H)
                img_H = util.augment_img(img_H, mode=mode)
                img_L = util.augment_img(img_L, mode=mode)
        else:
            img_H = self._modcrop(img_H)
            img_L = self._degrade(img_H)

        if bit_depth == 16:
            img_H = util.uint162single(img_H)
            img_L = util.uint162single(img_L)
        else:
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
        if self.opt['phase'] == 'train' and self.lr_noise_std > 0:
            noise = np.random.normal(0.0, self.lr_noise_std, img_L.shape).astype(np.float32)
            img_L = np.clip(img_L + noise, 0.0, 1.0)
        if self.opt['phase'] == 'train' and self.lr_mask_prob > 0:
            local_mean = cv2.blur(img_L, (3, 3), borderType=cv2.BORDER_REFLECT_101)
            if local_mean.ndim == 2:
                local_mean = np.expand_dims(local_mean, axis=2)
            replacement_noise = np.random.normal(
                0.0, self.lr_mask_noise_std, img_L.shape
            ).astype(np.float32)
            replacement = np.clip(local_mean + replacement_noise, 0.0, 1.0)
            mask = np.random.random(img_L.shape[:2]) < self.lr_mask_prob
            img_L = np.where(mask[..., None], replacement, img_L)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        return {
            'L': img_L,
            'H': img_H,
            'L_path': H_path,
            'H_path': H_path,
            'group': sample['group'],
            'sample_id': sample['sample_id'],
            'save_preview': sample['save_preview'],
            'bit_depth': bit_depth,
        }

    def __len__(self):
        return len(self.paths_H)
