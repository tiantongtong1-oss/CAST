# Disgust pseudo-label audit and ablation

Based on experiment/source-global-gaussian-knn-reliability-v7. Class IDs and the
original adaptive threshold formula remain unchanged; default training selection
remains confidence/KNN OR. Target truth is used only for diagnostic matrices.

## Run

Set CKPT to the existing source checkpoint, and preserve all other original
experiment arguments (including paths, backbone, epochs and KNN bandwidth).

```bash
python train.py --pre_epochs 0 --checkpoint "$CKPT" --audit_only --run_name audit
python train.py --pre_epochs 0 --checkpoint "$CKPT" --run_name baseline > baseline.log 2>&1
python train.py --pre_epochs 0 --checkpoint "$CKPT" --exclude_target_disgust --run_name exclude_disgust > exclude_disgust.log 2>&1
```

Outputs are isolated under models/rafdb_fer/<run_name>; the loaded source checkpoint
is not overwritten when pre_epochs=0. Keep threshold_beta=0.5 for this ablation.
Test threshold_beta=0 separately, using another run_name.

## Diagnostics

- PRE-TARGET: deterministic center-crop single-view prediction on all target train
  samples, before any target optimizer update. Confidence thresholds are computed
  on that same deterministic view. RNG states are restored afterwards.
- Each training epoch: online dual-view all/confidence/final/training matrices,
  plus confidence_AND_KNN and rescue when KNN is enabled. AND is diagnostic only;
  it does not replace the OR selection policy.
- Every validation/test call: full confusion counts, per-class precision/recall
  and macro-F1. Existing best-checkpoint selection still uses validation accuracy.

Matrices have rows=true, columns=predicted. selected_recall uses true counts inside
the selected subset; accepted_recall divides correct accepted predictions by all
true samples in the audited pass. Compare full validation recall/F1 for model
quality; a reduction in predicted disgust alone is not sufficient improvement.
NA indicates no predictions in a class. All seven classes contribute to macro-F1.

## Exclusion scope

--exclude_target_disgust clears target masks for pseudo class 2 after OR selection.
The same training mask controls classification, affinity class grouping, prototype
loss and target prototype updates. All source classes retain genuine supervision.
Final selection counts remain pre-exclusion; the training audit reports actual
post-exclusion counts. Excluded targets still participate in the forward pass/BN
and may remain in KNN memory. This isolates direct pseudo-label losses/updates.

The GitHub base does not include the local log's Tightening_Floor schedule or
PSEUDO AUDIT implementation. This change does not recreate that unprovided code.
Use identical implementations/configurations for both experiment runs.

## Verification

```bash
python -m unittest discover -s tests -p test_pseudo_audit.py
python -m py_compile train.py pseudo_audit.py
```

Full GPU training and dataset/checkpoint validation must be run on the training
server. Review baseline versus exclusion validation disgust precision, recall,
macro-F1 and accuracy, then repeat promising comparisons across random seeds.

## Export diagnostic images

Run with `--audit_only --export_audit_images --run_name source_images` and the
source checkpoint. Repeat using the exclusion target-best checkpoint and
`--run_name excluded_images`. Default outputs: image_audit/<run_name>.
An existing output directory is rejected; use a new --audit_image_dir to repeat.
Each group (angry_to_disgust, disgust_correct, disgust_to_sad, disgust_to_angry)
contains all matching records in samples.csv, plus up to 30 high-confidence and
30 random remaining original images. CSV IDs identify the same FER train paths
across checkpoint runs; confidence_accepted records fixed-view threshold passage.
metadata.json records checkpoint and inference-view details. Images are originals,
not augmented/model-input crops. Target truth is diagnostic only. Export is opt-in
and cannot run during training. These are student predictions, not EMA predictions.

## Source-only KNN and probability audit

Add `--audit_source_knn --audit_source_knn_k 20` to image-export audit commands.
Use a NEW run_name, e.g. source_knn_images. The loaded student checkpoint builds
both RAFDB and FER features with fixed views and the same pre-classifier feature
layer. The source bank contains only RAFDB train genuine labels, never target
labels or target pseudo-labels. Features are L2-normalized; neighbors use cosine
similarity and unweighted votes. CPU queries use chunks of at most 64 targets.

Group samples.csv now includes all seven p_* columns, source_support_* columns,
source_knn_pred/source_knn_agrees/source_mean_neighbor_cosine. balanced_source_*
columns repeat the query against a reproducible equal-count-per-class source
subset (seed 2000, minimum source class size). Summary fractions are computed
on ALL group members, not only copied images. summary.json reports per-group
angry/disgust prediction fractions, agreement fractions, and median disgust
support. Empty groups have count=0 and no rates. neighbors.csv lists the raw-bank
nearest source paths, labels and similarities for every matching target sample.
Metadata records source counts, k, and balanced subset size. Vote ties use the
lowest class ID; support is a vote fraction, not a calibrated probability.
No additional acceptance gate is applied: agreement is diagnostic only. Compare
angry_to_disgust rejection against disgust_correct retention before implementing
any training gate. Full CUDA extraction requires validation on the training server.
