#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

cast_checkpoint="${CAST_CHECKPOINT:-./models/rafdb_fer/resnet50_rafdb_fer_0.5765.pth}"
if [[ ! -f "$cast_checkpoint" ]]; then
  echo "Checkpoint not found: $cast_checkpoint" >&2
  exit 1
fi

cast_run="R50_continue_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p logs

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u train.py \
  --backbone resnet50 \
  --source_checkpoint "$cast_checkpoint" \
  --epochs 30 \
  --w2 0 \
  --target_w2 0.03 \
  --w3 0.1 \
  --ema_decay 0.995 \
  --target_lr 0.0002 \
  --run_name "$cast_run" \
  "$@" 2>&1 | tee "logs/${cast_run}.log"
