#!/usr/bin/env bash
set -euo pipefail

cast_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$cast_project_root"

# This exact source checkpoint was saved in the user's v4 training log.
cast_checkpoint="${CAST_SOURCE_CHECKPOINT:-./models/cast_resnet50_v4/resnet50_rafdb_fer_source_final.pth}"
if [[ ! -f "$cast_checkpoint" ]]; then
  printf 'Source checkpoint not found: %s\nSet CAST_SOURCE_CHECKPOINT to its actual path.\n' "$cast_checkpoint" >&2
  exit 1
fi

cast_run_id="$(date +%Y%m%d_%H%M%S)_$$"
cast_output="${CAST_MODEL_DIR:-./models/cast_resnet50_v5/$cast_run_id}"
if [[ -e "$cast_output" ]]; then
  printf 'Output already exists: %s\nChoose a new CAST_MODEL_DIR.\n' "$cast_output" >&2
  exit 1
fi
mkdir -p "$cast_output"

python -u train.py \
  --backbone resnet50 \
  --checkpoint "$cast_checkpoint" \
  --source_root "${CAST_SOURCE_ROOT:-/workspace/ttt/code/test-upload-clean/datesets/raf-basic}" \
  --target_root "${CAST_TARGET_ROOT:-/workspace/ttt/code/data/fer2013}" \
  --model_dir "$cast_output" \
  --epochs 30 --batch_size 64 --workers 10 \
  --bn_mode domain --teacher_temperature 0 \
  --threshold_mode quantile --augmentation face \
  --recovery_topk_per_class 0 --target_w2 0 \
  "$@" 2>&1 | tee "$cast_output/train.log"
