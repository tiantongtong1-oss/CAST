#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

cast_source_checkpoint="${CAST_SOURCE_CHECKPOINT:-./models/rafdb_fer/resnet50_rafdb_fer_source_best.pth}"
if [[ ! -f "$cast_source_checkpoint" ]]; then
  echo "Source checkpoint not found: $cast_source_checkpoint" >&2
  echo "Set CAST_SOURCE_CHECKPOINT to an existing source checkpoint, or use the full-training command in README.md." >&2
  exit 1
fi
cast_run_name="R50_stable_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p logs
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u train.py \
  --backbone resnet50 \
  --source_checkpoint "$cast_source_checkpoint" \
  --epochs 30 --w2 0 --target_w2 0 --w3 0.1 \
  --target_lr 0.0003 --ema_decay 0.995 \
  --teacher_views weak --threshold_min 0.8 --threshold_max 0.95 \
  --target_loss_reduction normalized --target_cls_weight 0.5 --pseudo_ramp 5 \
  --seed 1314 --run_name "$cast_run_name" "$@" \
  2>&1 | tee "logs/${cast_run_name}.log"
