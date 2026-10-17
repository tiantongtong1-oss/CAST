# Source-only DS with source class-weighted classification (v18)

Based on experiment/source-only-ds-radius-v12 at 5251a28a7d27c2d35c128bd509eef6b41573216f. This branch now also supports scheduled JS rescue filtering; see the section below. No target-anchor-only mechanism or balanced sampler is added.

## Enable

--source_class_weight_power 0.5 uses w_c proportional to 1/sqrt(n_c), where n_c counts only source training labels. Weights are normalized to mean 1 over present classes; missing classes get zero and cannot occur as a source training label. Default power 0 disables weighting and restores the original classification reduction. Powers in [0,1] are accepted; 1 means inverse frequency, which is stronger than the recommended initial experiment.

Source pretraining (if requested) uses sum(w_y * CE_s)/sum(w_y). During target self-training, with Ns source examples and Nt accepted target examples:

L_cls = [Ns * sum(w_y * CE_s)/sum(w_y) + sum(mask_t * CE_t)] / (Ns + Nt).

This normalizes weights within the source batch and retains source/target mixture coefficients Ns/(Ns+Nt) and Nt/(Ns+Nt). Accepted target samples are not class-weighted. Rejected targets retain zero classification weight. Source labels are the only input to class-count weights; target ground truth, target pseudo-label frequencies and validation labels are not used. Uniform weights recover the original masked mean.

Feature affinity, classifier modulation, prototype consistency, DS acceptance, thresholds, anchor promotion, sampling and immediate prototype update behavior are unchanged. Classification gradients and subsequent model/Teacher predictions will change as intended. Evaluation CE remains unweighted for comparable reporting.

## Run

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --pre_epochs 0 --epochs 30 \
  --source_class_weight_power 0.5 \
  --rescue_js_threshold 0.01 --rescue_js_start_epoch 20 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --immediate_prototype_update \
  --ds_neighbor --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 --ds_strength_cap 0.95 \
  --ds_support_threshold 0.6 --ds_margin_threshold 0.1 \
  --ds_uncertainty_threshold 0.3 --ds_conflict_threshold 0.5 \
  --run_name source_ds_q90_k11_srcsqrt_js001_start20_immediate \
  2>&1 | tee test_logs/source_ds_q90_k11_srcsqrt_js001_start20_immediate.log
```

With --pre_epochs 0 the original source checkpoint is unchanged; weighting begins on source examples used during target adaptation. This does not retroactively rebalance source pretraining. For an unweighted control omit --source_class_weight_power or set it to 0, changing run_name. Do not reuse a run name if prior outputs are needed.

Startup prints Source_Class_Weight_Power, Source_Class_Counts and Source_CE_Weights. The existing source-only DS diagnostic CSV schema is unchanged; output files use the new run_name.

## Validation

```bash
python -m unittest discover -s tests -v
python -m py_compile train.py source_class_weighting.py
```

Checks cover weight ratios/normalization, missing classes, invalid counts/powers, source reduction, target coefficient preservation, rejected targets, zero accepted targets, uniform/no-weight equivalence and CLI wiring. A PyTorch gradient test is skipped when torch is unavailable. Full data training was not executed in the authoring environment; accuracy improvements are not guaranteed.


## Scheduled JS addition on the same v18 branch

--rescue_js_threshold 0.01 --rescue_js_start_epoch 20 enables the extra natural-log JS criterion for rescue only at target epochs 20 onward (zero-based). Epochs 0-19 do not compute or filter JS; source class-weighted CE is active throughout target adaptation when --source_class_weight_power 0.5 is set. High-confidence direct acceptance is unchanged. Omit --rescue_js_threshold for the weighted no-JS control; start_epoch defaults to 0 for callers that explicitly want all-epoch JS.

DS audits all radius candidates; ds_pass is the DS-only outcome, js_pass is the JS outcome, accepted requires both. Before activation js_enabled=False, js_divergence is blank, js_pass=True, and accepted equals ds_pass. Logs show JS_Enabled and JS_Start_Epoch every epoch and JS_Rejected_After_DS as the additional rejection count. No target ground truth is used. Model/prototype changes caused by reweighted losses and filtered supervision are expected; no gain is guaranteed.

This experiment restarts full 30-epoch target training from the supplied source checkpoint rather than resuming epoch 20. Use a distinct run_name for comparison. Mathematical JS code matches the earlier offline diagnostic implementation (range [0, ln(2)], not square-root distance).
