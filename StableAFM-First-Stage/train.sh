#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAIR_ROOT="${AFM_SWINIR_KAIR:-$PROJECT_ROOT/KAIR}"
CONFIG="${1:-$PROJECT_ROOT/configs/finetune_afm_x4.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
MASTER_PORT="${MASTER_PORT:-29632}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PWD/$CONFIG"
fi

if [[ ! -f "$KAIR_ROOT/main_train_psnr.py" ]]; then
  echo "KAIR is missing at $KAIR_ROOT; run install_overlay.sh first." >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config is missing: $CONFIG" >&2
  exit 1
fi

cd "$KAIR_ROOT"
exec "$PYTHON_BIN" -m torch.distributed.run \
  --nproc-per-node="$NPROC_PER_NODE" \
  --master-port="$MASTER_PORT" \
  main_train_psnr.py \
  --opt "$CONFIG" \
  --dist true
