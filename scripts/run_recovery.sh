#!/usr/bin/env bash
set -euo pipefail

# Reuse the source checkpoint from the supplied best prototype experiment.
# A different path or seed can be supplied via the trailing CLI arguments.
bash "$(dirname "${BASH_SOURCE[0]}")/run_temporal_consistency.sh" \
  --pre_epochs 0 \
  --checkpoint models/rafdb_fer/mobilenet_v2_rafdb_fer_prototype_consistency_v1_source_best.pth \
  "$@"
