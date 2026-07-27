# StableAFM

中文 | [English](README.md)

论文 **《A Physics-Guided Generative Diffusion Model for Super-Resolution AFM
of Trap Dynamic Behavior》** 的官方实现。

## 仓库结构

```text
StableAFM/
├── StableAFM-First-Stage/     # AFM-SwinIR x4 超分辨率
└── StableAFM-Second-Stage/    # StableAFM 扩散重建
```

本仓库以覆盖层形式提供基于固定版本 KAIR 和 StableSR 的修改代码。运行每个阶段
前，请先按照对应上游项目的说明配置环境。

## 第一阶段：AFM-SwinIR

### 安装

```bash
cd StableAFM-First-Stage
git clone https://github.com/cszn/KAIR.git KAIR
git -C KAIR checkout fc1732f4a4514e42ce15e5b3a1e18c828af47a1e
bash install_overlay.sh
```

### 训练

先在 `configs/` 中修改数据路径和运行参数，然后执行：

```bash
bash train.sh configs/pretrain_df2k_x4.json
bash train.sh configs/finetune_afm_x4.json
```

### 推理

```bash
python tools/infer_afm.py \
  --kair-root KAIR \
  --input /path/to/input.npy \
  --output-dir output/sample \
  --config configs/finetune_afm_x4.json \
  --checkpoint checkpoints/StableAFM-firststage-AFM-SwinIR.pth \
  --device cuda:0
```

## 第二阶段：StableAFM

### 安装

```bash
cd ../StableAFM-Second-Stage
git clone https://github.com/IceClear/StableSR.git StableSR
git -C StableSR checkout 398ee9383777e255540ea027a704c8ce1f32145b
bash install_overlay.sh
```

### 训练

先在 `configs/train_afm.yaml` 中修改数据路径和运行参数，然后执行：

```bash
bash train.sh StableSR configs/stableafm/train_afm.yaml 0,1
```

### 推理

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

## 模型权重

第一阶段推理权重位于
`StableAFM-First-Stage/checkpoints/StableAFM-firststage-AFM-SwinIR.pth`。
第二阶段权重后续通过 Google Drive 提供。
