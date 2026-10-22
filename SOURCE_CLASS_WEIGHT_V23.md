# Source-only class weighting (v23)

Base: experiment/radius-or-knn-vote6of11-v22, ad53a5f053d5cbf6694dfcba74de872e9108a59a.
Branch: experiment/source-class-weight-sqrt-v23.

Enable with --source_class_weight inverse_power. Default none preserves unweighted CE.
Counts come only from source_train.label. Present-class weights are n**(-power),
normalized to class mean 1, then capped at max (no second normalization).
Defaults: power=0.5, max=2.0. Absent classes get zero in enabled mode.
CE uses sum(weight * loss) / sum(weight). In the mixed target-stage batch, only
source rows receive class weights; accepted target rows retain their existing
confidence masks. The joint denominator includes these source weights and target
masks. Source/target relative contribution can consequently change. Evaluation CE,
source sampling, affinity/anchor masks, prototypes, EMA, thresholds and 6/11 rescue
are unchanged. This is a loss-weighting ablation, not balanced sampling.

Works with pre_epochs=0: the labeled source rows used during target adaptation
are still weighted. Checkpoint loads model weights, not optimizer/scheduler resume.
Use the SAME source checkpoint as v22 for this controlled comparison.

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 0 --epochs 30 --lr 0.001 \
  --source_class_weight inverse_power \
  --source_class_weight_power 0.5 --source_class_weight_max 2.0 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --radius_rescue_warmup_epochs 3 --immediate_prototype_update \
  --ds_k 11 \
  --run_name adaptive_minconf_radius_or_vote6of11_source_sqrt_v23 \
  2>&1 | tee test_logs/adaptive_minconf_radius_or_vote6of11_source_sqrt_v23.log
```

For a disabled control, set --source_class_weight none and use another run_name.
To also retrain source from the same initialization use --pre_epochs 30; this is
an additional experiment rather than the same-checkpoint target-only ablation.

Validation: 4 weight tests + 4 neighbor vote tests; syntax and diff checks pass.
No PyTorch/CUDA or RAF/FER data available in the editing environment; full training
and tensor integration have not been run. No accuracy improvement is claimed.
