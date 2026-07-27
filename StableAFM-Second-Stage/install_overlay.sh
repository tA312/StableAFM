#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLESR_ROOT="${1:-$PROJECT_ROOT/StableSR}"
BASE_COMMIT="398ee9383777e255540ea027a704c8ce1f32145b"

if [[ ! -f "$STABLESR_ROOT/main.py" ||
      ! -f "$STABLESR_ROOT/ldm/models/diffusion/ddpm.py" ]]; then
  echo "Not a StableSR checkout: $STABLESR_ROOT" >&2
  exit 1
fi

if [[ -d "$STABLESR_ROOT/.git" ]]; then
  current_commit="$(git -C "$STABLESR_ROOT" rev-parse HEAD)"
  if [[ "$current_commit" != "$BASE_COMMIT" ]]; then
    echo "warning: overlay was prepared for StableSR $BASE_COMMIT" >&2
    echo "warning: target checkout is $current_commit" >&2
  fi
fi

OVERLAY_FILES=(
  main.py
  basicsr/data/degradations.py
  basicsr/data/grayscale_swinir_paired_dataset.py
  ldm/models/diffusion/ddpm.py
  ldm/models/diffusion/latent_dps.py
)

for relative_path in "${OVERLAY_FILES[@]}"; do
  source_path="$PROJECT_ROOT/overlay/$relative_path"
  target_path="$STABLESR_ROOT/$relative_path"
  backup_path="$target_path.original"
  mkdir -p "$(dirname "$target_path")"
  if [[ -f "$target_path" && ! -e "$backup_path" ]]; then
    cp -p "$target_path" "$backup_path"
  fi
  cp -p "$source_path" "$target_path"
  echo "installed $relative_path"
done

mkdir -p "$STABLESR_ROOT/configs/stableafm" "$STABLESR_ROOT/scripts" \
  "$STABLESR_ROOT/tests"
cp -p "$PROJECT_ROOT/configs/train_afm.yaml" \
  "$STABLESR_ROOT/configs/stableafm/train_afm.yaml"
cp -p "$PROJECT_ROOT/tools/infer_afm.py" \
  "$STABLESR_ROOT/scripts/infer_afm.py"
cp -p "$PROJECT_ROOT/tests/test_latent_dps.py" \
  "$STABLESR_ROOT/tests/test_latent_dps.py"

echo "StableAFM overlay installed in $STABLESR_ROOT"
echo "Existing files were backed up once with the suffix .original"
