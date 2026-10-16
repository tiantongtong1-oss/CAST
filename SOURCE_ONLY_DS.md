# Source-only DS audit after radius rescue

Based on experiment/dual-domain-ds-radius-v11 at 4f37c74882d155de7f47a28d02cfabe8b0805975.
This branch makes --ds_neighbor use source evidence only. DUAL_DOMAIN_DS_V11.md describes the historical parent version, not this branch's active training path.

## Selection

- High-confidence dual-view agreement is accepted directly, without DS or a radius requirement.
- Other agreeing samples are rescue candidates. Rescue still requires its enabled flag and warmup, then strict membership in the source class quantile radius.
- With --ds_neighbor, those candidates additionally require source evidence availability, support >= ds_support_threshold, margin >= ds_margin_threshold, unknown <= ds_uncertainty_threshold, and source conflict <= ds_conflict_threshold.
- Without --ds_neighbor, keep the original radius rescue.

## Source reference bank

Refresh all labeled source features with the current eval EMA Teacher at each target epoch start, and freeze the bank for that epoch. Source quality is 1. Search across all source classes, within ds_max_distance, retaining at most ds_k neighbors. Source and target IDs are separate namespaces; do not exclude a source reference because its numeric ID matches a target query ID.

No target reference dataset, target anchor neighbor bank, sender_scores check, cross-domain pooling or cross-domain conflict is used in training. Historical dual-domain numerical helpers remain for compatibility/tests but are not called by train.py. --ds_source_weight is retained only as a compatibility argument and must equal 1.0 (default).

Anchor promotion, MMD anchor masks and the optional --immediate_prototype_update retain their previous behavior. A source-only audit does not require role separation; changing that flag still changes the baseline anchor behavior.

Diagnostics contain source_count, source_available, source_support, source_unknown, source_conflict, final support/margin/unknown and accepted. Target-domain and cross-domain fields are removed rather than filled with misleading zeros.

This change does not address evidence saturation from correlated neighbors and does not promise accuracy gains.

## Example

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --pre_epochs 0 --epochs 30 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --immediate_prototype_update \
  --ds_neighbor --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 \
  --ds_support_threshold 0.6 --ds_margin_threshold 0.1 \
  --ds_uncertainty_threshold 0.3 --ds_conflict_threshold 0.5 \
  --ds_strength_cap 0.95 \
  --run_name source_only_ds_q90_k11_immediate \
  2>&1 | tee test_logs/source_only_ds_q90_k11_immediate.log
```

For a matched no-DS control, remove --ds_neighbor and change --run_name. Keep --immediate_prototype_update and all other settings identical. Extra source reference extraction may consume random state, so identical global seeds do not guarantee identical batch/augmentation sequences between DS on/off runs.

Run names overwrite their corresponding outputs if reused. Checkpoints initialize model weights; this command is not exact training resumption.

## Checks

```bash
python -m unittest discover -s tests -p 'test_*ds.py' -v
python -m py_compile train.py dual_domain_ds.py
```

Numerical, gate and training-wiring checks run without torch/data. Full training requires the RAF/FER datasets and PyTorch/CUDA environment.
