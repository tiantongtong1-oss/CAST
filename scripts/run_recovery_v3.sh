#!/usr/bin/env bash
set -euo pipefail

# Recovery v3 keeps the reproducible recovery-v2 data/RNG setup, but turns on
# two conservative regularizers suggested by the recovery_base_s1314 log:
#   1) bounded source-class balancing to strengthen clean minority supervision;
#   2) a small temporal soft-label component to reduce over-confident hard
#      pseudo-label fitting without enabling the aggressive temporal hard gate.
# Extra CLI arguments are appended last and therefore override these defaults.
bash "$(dirname "${BASH_SOURCE[0]}")/run_temporal_consistency.sh" \
  --pre_epochs 0 \
  --checkpoint models/rafdb_fer/mobilenet_v2_rafdb_fer_prototype_consistency_v1_source_best.pth \
  --temporal_mode observe \
  --target_soft_weight 0.20 \
  --source_balance_power 0.50 \
  --source_balance_max_ratio 2.0 \
  "$@"
