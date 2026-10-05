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
