#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
run_tag="prototype_recovery_v2_$(date -u +%Y%m%d_%H%M%S)"

# Recovery defaults: observe temporal history without filtering, hard target
# CE, unchanged prototype objective. --temporal_mode filter is now opt-in.
# Extra CLI arguments override defaults, including paths, seed and run_name.
python train.py \
  --backbone mobilenet_v2 \
  --pre_epochs 30 \
  --epochs 30 \
  --seed 1314 \
  --lr 0.001 \
  --w1 4 --w2 0.3 --w3 0.1 \
  --ema_decay 0.999 \
  --threshold_base 0.85 --threshold_beta 0.5 --threshold_margin 0.02 \
  --threshold_min 0.80 --threshold_max 0.95 \
  --proto_weight 0.10 --proto_temperature 0.20 --proto_momentum 0.99 \
  --proto_source_anchor 0.50 --proto_warmup_epochs 3 --proto_ramp_epochs 5 \
  --temporal_momentum 0.7 --temporal_min_streak 2 --temporal_warmup_epochs 2 \
  --target_soft_weight 0 \
  --eval_ema \
  --run_name "$run_tag" \
  "$@" \
  2>&1 | tee "logs/${run_tag}.log"
