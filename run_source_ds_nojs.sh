#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p test_logs
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --pre_epochs 0 --epochs 30 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --immediate_prototype_update \
  --ds_neighbor --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 --ds_strength_cap 0.95 \
  --ds_support_threshold 0.6 --ds_margin_threshold 0.1 \
  --ds_uncertainty_threshold 0.3 --ds_conflict_threshold 0.5 \
  --no_rescue_js \
  --run_name source_ds_q90_k11_nojs_immediate \
  2>&1 | tee test_logs/source_ds_q90_k11_nojs_immediate.log
