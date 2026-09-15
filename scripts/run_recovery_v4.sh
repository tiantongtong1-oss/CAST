#!/usr/bin/env bash
set -euo pipefail

# Recovery v4 addresses the two failure modes visible in recovery_v3_s1314:
#   1) target pseudo-label starvation for fear/disgust/angry;
#   2) late-stage over-confidence / validation-selection variance.
#
# The new target balancing is deliberately mild and uses pseudo labels only.
# Compared with v3, soft CE and prototype regularization are reduced slightly so
# the rebalanced target CE can affect the student without stacking too many
# regularizers at full strength.
export CAST_TARGET_BALANCE_POWER="${CAST_TARGET_BALANCE_POWER:-0.20}"
export CAST_TARGET_BALANCE_MAX_RATIO="${CAST_TARGET_BALANCE_MAX_RATIO:-1.50}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
run_tag="recovery_v4_$(date -u +%Y%m%d_%H%M%S)"

python recovery_v4.py \
  --backbone mobilenet_v2 \
  --pre_epochs 0 \
  --checkpoint models/rafdb_fer/mobilenet_v2_rafdb_fer_prototype_consistency_v1_source_best.pth \
  --epochs 30 \
  --seed 1314 \
  --lr 0.001 \
  --w1 4 --w2 0.3 --w3 0.1 \
  --ema_decay 0.999 \
  --threshold_base 0.85 --threshold_beta 0.5 --threshold_margin 0.02 \
  --threshold_min 0.80 --threshold_max 0.95 \
  --proto_weight 0.05 --proto_temperature 0.20 --proto_momentum 0.99 \
  --proto_source_anchor 0.50 --proto_warmup_epochs 3 --proto_ramp_epochs 5 \
  --temporal_mode observe \
  --temporal_momentum 0.7 --temporal_min_streak 2 --temporal_warmup_epochs 2 \
  --target_soft_weight 0.10 \
  --source_balance_power 0.50 --source_balance_max_ratio 2.0 \
  --eval_ema \
  --run_name "$run_tag" \
  "$@" \
  2>&1 | tee "logs/${run_tag}.log"
