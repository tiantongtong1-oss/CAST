# Target-anchor-only DS radius rescue (v14)

Based on experiment/source-only-ds-diagnostics-v13. Train with --ds_neighbor to use ONLY target anchors as neighbor evidence. Source labels still define the source prototype/radius gate; there is no source-neighbor evidence bank or cross-domain pool.

## Anchor lifecycle

1. A target sample is eligible only if both weak views agree, both meet the predicted class adaptive confidence threshold, and the fused feature lies strictly inside its source class radius.
2. Only consecutive eligible epochs with the same label count toward promotion (default 3). A rescued-only sample cannot accumulate a streak or become a DS sender. Same-epoch repeated observations do not increase the streak.
3. At epoch start, first compute current thresholds and refresh source radii when scheduled. Re-extract prior promoted anchors with current eval EMA Teacher and deterministic test transform. Require the current prediction to match its historical label, current confidence to meet the current class threshold, and current feature to pass the source radius. This is a single deterministic-view revalidation; historical promotion used both weak views.
4. Failed revalidation excludes the sender and resets its streak and quality. It must qualify again. Retained sender quality is min(current confidence, previous recorded dual-view minimum confidence).
5. Freeze the resulting bank for the whole epoch. Query IDs exclude themselves. Fewer than ds_min_neighbors (after exclusion and distance filtering), excessive within-target conflict or failing support/margin/unknown thresholds prevents rescue. Empty banks return vacuous evidence, while high-confidence acceptance continues normally.

High-confidence acceptance does not require source radius or DS. The radius is required for rescue AND for anchor promotion. final_mask remains confidence_mask | rescue_mask. With role separation enabled, anchor_mask for MMD and default prototype writes now also uses high-confidence-only eligibility. --immediate_prototype_update still permits all accepted samples to update target prototypes; it does NOT grant them sender eligibility. Rescued training samples can still indirectly affect model/feature learning: this design blocks direct rescued-label enrollment, not all feedback.

--ds_neighbor requires role separation. --ds_source_weight is retained as a legacy unused option; it does not pool source evidence. Without --ds_neighbor, radius rescue runs without neighbor auditing, but the new high-confidence anchor promotion rule remains. For a matched control use this branch, omit --ds_neighbor and change run_name.

## Run

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --pre_epochs 0 --epochs 30 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --promotion_epochs 3 --radius_rescue_warmup_epochs 3 \
  --immediate_prototype_update \
  --ds_neighbor --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 \
  --ds_support_threshold 0.6 --ds_margin_threshold 0.1 \
  --ds_uncertainty_threshold 0.3 --ds_conflict_threshold 0.5 \
  --ds_strength_cap 0.95 \
  --run_name target_anchor_ds_q90_k11_highconf_immediate \
  2>&1 | tee test_logs/target_anchor_ds_q90_k11_highconf_immediate.log
```

This starts a new training experiment from source weights. It is NOT an offline diagnostic or exact resume. Initial target banks are empty; with promotion_epochs=3, first senders can appear at epoch 3 (epochs numbered from 0). Per-class anchor imbalance and persistent wrong predictions remain possible; no accuracy gain is promised. Reusing a run name overwrites corresponding outputs.

## Outputs and compatibility

Startup prints Evidence: target_anchor_only and Anchor_Eligibility: high_confidence_and_source_radius. Each epoch prints DS_Target_Anchor_Refs and DS_Refresh_Revoked. The DS CSV uses target_count, target_available, target_support, target_unknown, target_conflict, support, margin, unknown, accepted; source/cross-domain fields are absent.

Historical SOURCE_ONLY_DS.md, DUAL_DOMAIN_DS_V11.md and diagnose_ds.py describe source-only/dual-domain experiments. The inherited diagnose_ds.py intentionally remains a SOURCE reference diagnostic and does not reproduce this target-anchor audit. Do not interpret its output as this branch's neighbor decisions. Target history is in training checkpoints, but train.py continues weights-initialization-only behavior.

## Checks

```bash
python -m unittest discover -s tests -p '*ds*.py' -v
python -m py_compile train.py dual_domain_ds.py target_anchor_ds.py
```

Numerical evidence, refresh-gate, self-exclusion, empty-bank, frozen-copy and AST wiring checks run without torch. PyTorch tests also exercise promotion/reset and the actual refresh function with synthetic CPU data, and are skipped when torch is absent. The authoring environment has no torch/CUDA/RAF/FER; full training was not executed.
