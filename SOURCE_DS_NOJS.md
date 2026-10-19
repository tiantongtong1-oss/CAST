# Source DS without JS (v18)

Base: experiment/source-ds-js-late-v17 at 13a46860368cb4efbfd1fb1bbedfd04f2d8cf2a1.

Run `bash run_source_ds_nojs.sh` from the repository with the same dependencies, datasets and source checkpoint as v17. The script retrains 30 target epochs from the same source checkpoint; it is not an exact resume. Paths match SOURCE_DS_JS_RESCUE.md; edit them for your machine if needed.

`--no_rescue_js` overrides any supplied JS threshold and disables JS computation and filtering for all target epochs, including epoch 20 onward. Source-only DS, radius rescue, dual-view agreement/high-confidence acceptance, EMA, anchor promotion, immediate prototype updates, losses and random seed keep their v17 behavior. JS_Enabled must remain False; CSV js_divergence is blank, js_pass=True and accepted equals ds_pass.

The run name is source_ds_q90_k11_nojs_immediate to avoid replacing v17 outputs. Compare it with v17 using the same data and source checkpoint.

Validation: six schedule/CLI/wiring tests passed; Python compilation and shell syntax passed. Full RAF/FER training was not run in the authoring environment.

```bash
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
```
