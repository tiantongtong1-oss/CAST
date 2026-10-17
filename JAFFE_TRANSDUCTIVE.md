# RAF-DB -> JAFFE full-pool transductive experiment (v19)

Based on `experiment/source-ds-js-late-v17`, commit `13a4686`.
No source classification reweighting is added. Source DS, dual-view selection,
prototype/anchor updates, and epoch-20 rescue-only JS retain v17 behavior.

## Data and protocol

Point `--target_path` at the original JAFFE directory, including nested directories.
Original TIFF filenames such as `KA.AN1.39.tiff` and pose IDs such as `KA-AN1.tif`
are supported. Readable PNG/JPEG conversions retaining pose names are also supported.
Unknown image filenames and duplicate subject/pose/number records fail explicitly.
The original pool is 213 images; a count mismatch prints a warning, not an invented split.

Filename pose mapping to CAST: SU=0, FE=1, DI=2, HA=3, SA=4, AN=5, NE=6.
Use the requested filename pose, not an argmax of the README semantic ratings.
TIFF grayscale is decoded as three-channel RGB for the existing transforms.
This code does not run RetinaFace or claim the supplied original images are newly aligned.
JAFFE README and images must remain local; no dataset assets are committed.

All image paths are shared by adaptation, threshold, and evaluation views. Adaptation
returns -1 in place of true labels. Training selection, MMD anchors and prototype writes
use model pseudo-labels only. Only the final deterministic evaluation view exposes labels.
There is no JAFFE validation dataset, per-epoch target accuracy evaluation, early stopping,
or target-best selection. Require `--pre_epochs 0` plus a source-only checkpoint. The FER
source checkpoint path below is reused as initialization, not its FER-adapted target model.

Predeclare 30 epochs; save the last student and EMA teacher together in `*_target_last.pth`.
Final metrics use the **student** (`model`), consistent with v17 final FER evaluation.
Do not compare student and EMA using JAFFE truth to pick the better model. The checkpoint
records `evaluation_protocol`; `best_val_acc` is null for JAFFE.
FER keeps its original validation-best selection.

At batch 128 and 213 target images, each epoch has 2 updates (60 for 30 epochs), including
the final 85-image batch (`drop_last=False`). Each target batch consumes one shuffled
source batch. Source iterator resets only if exhausted. Counts are printed at startup.
This deliberately preserves one target-pool pass per epoch; it is not FER-equivalent
optimization exposure. Warmups, LR decay and epoch-20 JS are unchanged, not retuned.

This is a declared transductive protocol, not a verified reconstruction of CAST's
unpublished JAFFE split. Same target images are adapted without labels and evaluated
with labels at the end. Final labels must not be used for parameter/epoch tuning.

## Run

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --data1 rafdb --data2 jaffe --backbone mobilenet_v2 \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/jaffe \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --pre_epochs 0 --epochs 30 --workers 0 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --immediate_prototype_update \
  --ds_neighbor --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 --ds_strength_cap 0.95 \
  --ds_support_threshold 0.6 --ds_margin_threshold 0.1 \
  --ds_uncertainty_threshold 0.3 --ds_conflict_threshold 0.5 \
  --rescue_js_threshold 0.01 --rescue_js_start_epoch 20 \
  --run_name jaffe_source_ds_js001_start20_immediate \
  2>&1 | tee test_logs/jaffe_source_ds_js001_start20_immediate.log
```

Output: `models/rafdb_jaffe/mobilenet_v2_rafdb_jaffe_jaffe_source_ds_js001_start20_immediate_target_last.pth`.

## Validation

`python -m unittest discover -s tests -p test_jaffe.py -v` exercises the actual
dataset class via isolated dependencies, label parsing, hidden adaptation labels,
shared IDs, three view tuples and malformed/duplicate inputs. Static compilation
also passes. Full torch/CUDA training and real JAFFE TIFF decoding were not executed
in the authoring environment (torch and the image dataset unavailable).
