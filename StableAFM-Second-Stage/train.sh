#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLESR_ROOT="${1:-$PROJECT_ROOT/StableSR}"
CONFIG_PATH="${2:-configs/stableafm/train_afm.yaml}"
VISIBLE_GPUS="${3:-${CUDA_VISIBLE_DEVICES:-0}}"
shift "$(( $# < 3 ? $# : 3 ))"

if [[ ! -f "$STABLESR_ROOT/main.py" ]]; then
  echo "Not a StableSR checkout: $STABLESR_ROOT" >&2
  exit 1
fi

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$STABLESR_ROOT/$CONFIG_PATH"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing training config: $CONFIG_PATH" >&2
  exit 1
fi

IFS=',' read -r -a gpu_array <<< "$VISIBLE_GPUS"
lightning_gpus=""
for index in "${!gpu_array[@]}"; do
  lightning_gpus+="$index,"
done

export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "$STABLESR_ROOT"
exec "${PYTHON_BIN:-python}" main.py \
  --train True \
  --no-test True \
  --base "$CONFIG_PATH" \
  --gpus "$lightning_gpus" \
  --name stableafm \
  --scale_lr False \
  "$@"
