# Source-only DS with dual-view JS rescue gate (v16)

Based on source-ds-js-diagnostics-v15; NOT target-anchor-only v14. Neighbor evidence remains source-only. Target anchor eligibility and prototype updates keep the source baseline behavior.

--rescue_js_threshold 0.01 enables an extra gate only on radius rescue. Both view softmax distributions are compared using natural-log Jensen-Shannon divergence (not square-root JS distance), in [0, ln(2)]. Rescue requires JS <= threshold in addition to agreement, low confidence, source radius, rescue enable/warmup and DS pass when --ds_neighbor is enabled. High-confidence acceptance is still confidence_mask OR rescue_mask and is unaffected by this gate. No target truth enters selection.

The default is None: omitting --rescue_js_threshold exactly disables the extra criterion. Keep other options identical for a no-JS control. With DS disabled, the JS option still gates radius rescue; it is not implicitly a DS toggle.

The 0.01 value is exploratory, selected from a final-checkpoint audit. No gain is promised, especially at early training epochs; validate on permitted independent data and use held-out tests only for final reporting.

## Command

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --pre_epochs 0 --epochs 30 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --immediate_prototype_update \
  --ds_neighbor --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 --ds_strength_cap 0.95 \
  --ds_support_threshold 0.6 --ds_margin_threshold 0.1 \
  --ds_uncertainty_threshold 0.3 --ds_conflict_threshold 0.5 \
  --rescue_js_threshold 0.01 \
  --run_name source_ds_q90_k11_js001_immediate \
  2>&1 | tee test_logs/source_ds_q90_k11_js001_immediate.log
```

New run_name avoids overwriting prior outputs. This is a new training experiment, not checkpoint diagnostic inference or exact resume.

## Diagnostics

All radius candidates still undergo DS and appear in its CSV, including those JS will reject. DS_Passed remains the count passing DS before JS. CSV adds js_divergence, js_pass and ds_pass; accepted now means BOTH DS and JS passed. With JS disabled its divergence column is blank and js_pass=True. Logs add JS_Radius_Candidates, JS_Passed (before DS), and JS_Rejected_After_DS. The last counter isolates samples that would have passed DS but are blocked by JS. Radius_Rescue_Num reports the final combined rescue count.

Training uses the same NumPy float64 JS helper as offline diagnostic analysis. This adds softmax-to-CPU transfers when enabled but does not change optimizer/model configuration. diagnose_ds.py continues to report ungated source DS decisions and JS values for offline analysis; it does not automatically apply this training option.

## Checks

```bash
python -m unittest discover -s tests -p '*ds*.py' -v
python -m unittest discover -s tests -p test_rescue_js.py -v
python -m py_compile train.py rescue_js.py
```

Numerical gate, CLI, boundary and wiring checks run without PyTorch. Full RAF/FER training was not executed in the authoring environment.
