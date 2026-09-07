# StableAFM

[中文](README.zh-CN.md) | English

Official implementation of **“A Physics-Guided Generative Diffusion Model for
Super-Resolution AFM of Trap Dynamic Behavior.”**

## Repository structure

```text
StableAFM/
├── StableAFM-First-Stage/     # AFM-SwinIR x4 super-resolution
└── StableAFM-Second-Stage/    # StableAFM diffusion reconstruction
```

The source is provided as overlays for pinned versions of KAIR and StableSR.
Install the corresponding upstream environment before running each stage.

## Stage 1: AFM-SwinIR

### Setup

```bash
cd StableAFM-First-Stage
git clone https://github.com/cszn/KAIR.git KAIR
git -C KAIR checkout fc1732f4a4514e42ce15e5b3a1e18c828af47a1e
bash install_overlay.sh
```

### Train

Update the dataset paths and runtime settings in `configs/`, then run:

```bash
bash train.sh configs/pretrain_df2k_x4.json
bash train.sh configs/finetune_afm_x4.json
```

### Inference

```bash
python tools/infer_afm.py \
  --kair-root KAIR \
  --input /path/to/input.npy \
  --output-dir output/sample \
  --config configs/finetune_afm_x4.json \
  --checkpoint checkpoints/StableAFM-firststage-AFM-SwinIR.pth \
  --device cuda:0
```

## Stage 2: StableAFM

### Setup

```bash
cd ../StableAFM-Second-Stage
git clone https://github.com/IceClear/StableSR.git StableSR
git -C StableSR checkout 398ee9383777e255540ea027a704c8ce1f32145b
bash install_overlay.sh
```

### Train

Update the dataset paths and runtime settings in `configs/train_afm.yaml`, then
run:

```bash
bash train.sh StableSR configs/stableafm/train_afm.yaml 0,1
```

### Inference

```bash
cd StableSR
python scripts/infer_afm.py \
  --input-dir /path/to/swinir_inputs \
  --measurement-dir /path/to/measurements \
  --config configs/stableafm/train_afm.yaml \
  --checkpoint /path/to/second_stage.ckpt \
  --output-dir outputs/inference \
  --sampler ddpm \
  --steps 200 \
  --dps-scale 10000 \
  --init-mode condition
```

## Model weights

The first-stage inference checkpoint is included at
`StableAFM-First-Stage/checkpoints/StableAFM-firststage-AFM-SwinIR.pth`.
The second-stage checkpoint
[`StableAFM-secondstage.ckpt`](https://drive.google.com/file/d/1gh1HJDfuwJiDPnyUvrnKClpKqVcRUNDZ/view?usp=sharing)
is available on Google Drive.
