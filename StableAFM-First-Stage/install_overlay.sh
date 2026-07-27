#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAIR_ROOT="${1:-$PROJECT_ROOT/KAIR}"

if [[ ! -f "$KAIR_ROOT/models/network_swinir.py" ]]; then
  echo "Not a KAIR checkout: $KAIR_ROOT" >&2
  exit 1
fi

FILES=(
  main_train_psnr.py
  data/select_dataset.py
  data/dataset_sr_decimation.py
  data/sampler_modality.py
  models/model_plain.py
  models/network_swinir.py
  utils/utils_image.py
  utils/utils_option.py
  utils/utils_sr_metrics.py
)

for relative_path in "${FILES[@]}"; do
  source_path="$PROJECT_ROOT/overlay/$relative_path"
  target_path="$KAIR_ROOT/$relative_path"
  backup_path="$target_path.original"
  mkdir -p "$(dirname "$target_path")"
  if [[ -f "$target_path" && ! -e "$backup_path" ]]; then
    cp -p "$target_path" "$backup_path"
  fi
  cp -p "$source_path" "$target_path"
  echo "installed $relative_path"
done

echo "AFM-SwinIR overlay installed in $KAIR_ROOT"
echo "Original tracked files were backed up once with the suffix .original"
