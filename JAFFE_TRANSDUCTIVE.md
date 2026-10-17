# RAF-DB -> JAFFE with RetinaFace preprocessing (v19)

Current branch: experiment/source-ds-js-jaffe-v19. Source DS, epoch-20 rescue-only
JS <= 0.01, anchor/prototype updates and unweighted classification remain unchanged.
FER validation-best behavior remains unchanged.

## Preprocess locally

Original JAFFE grayscale TIFF files stay unchanged. prepare_jaffe.py uses the
serengil/retinaface detector API, selects the largest detection (score tie-break),
and fits a deterministic five-point similarity transform to a standard 112-pixel
face template scaled to 256. It writes RGB-compatible PNGs and detection metadata.
This follows CAST's detection/alignment/256-size intent, but the precise alignment
implementation/template in CAST is not known; this is not an exact reproduction.
Dependency API reference: https://github.com/serengil/retinaface

Install the preprocessing dependencies in a separate venv, avoiding changes to
your existing PyTorch environment. Detector weights download automatically on first use.
No JAFFE images are committed or redistributed. Keep aligned copies local, following
JAFFE README usage conditions.

```bash
cd /workspace/ttt/code/CAST
python -m venv .venv_jaffe_align
.venv_jaffe_align/bin/python -m pip install --upgrade pip
.venv_jaffe_align/bin/python -m pip install 'retina-face==0.0.17' tf-keras
CUDA_VISIBLE_DEVICES='' .venv_jaffe_align/bin/python prepare_jaffe.py \
  --input /workspace/ttt/code/data/jaffe \
  --output /workspace/ttt/code/data/jaffe_retinaface_256
```

Output must be empty and separate from input. No detection failure falls back to
raw input. On any failure, inspect alignment_report.csv; the script exits nonzero,
and training also rejects a partial pool with failed metadata. Use a new empty output
for retries. Visually inspect local aligned faces for complete facial coverage before
running the experiment; do not upload/publish dataset images.

Filename pose labels: SU=0, FE=1, DI=2, HA=3, SA=4, AN=5, NE=6. Not semantic-rating
argmax. Original KA.AN1.39.tiff and KA-AN1.tif forms are accepted. Duplicate poses and
unknown image filenames are rejected. Adaptation returns -1 labels; evaluation alone
exposes labels. Stable IDs are shared across adaptation/threshold/evaluation views.

## Train without reading an experiment checkpoint

Run with the existing activated cast environment. Do not pass --checkpoint.
The default backbone still starts from torchvision ImageNet weights, just as before;
add --no_imagenet_pretrained only if intentional random initialization is desired.
RetinaFace also needs its detector weights. 'No checkpoint' here means no existing
CAST/FER checkpoint is read, not that external pretrained weights are forbidden.

```bash
cd /workspace/ttt/code/CAST
python -u train.py \
  --data1 rafdb --data2 jaffe --backbone mobilenet_v2 \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/jaffe_retinaface_256 \
  --pre_epochs 30 --epochs 30 --workers 0 \
  --model_dir /workspace/ttt/code/CAST/jaffe_models \
  --log_dir /workspace/ttt/code/CAST/jaffe_logs \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --immediate_prototype_update \
  --ds_neighbor --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 --ds_strength_cap 0.95 \
  --ds_support_threshold 0.6 --ds_margin_threshold 0.1 \
  --ds_uncertainty_threshold 0.3 --ds_conflict_threshold 0.5 \
  --rescue_js_threshold 0.01 --rescue_js_start_epoch 20 \
  --run_name jaffe_retinaface_source30_target30_js20
```

The source final weights stay in memory between stages. Save *_source_last.pth,
reset Adam and LR schedule for target adaptation, then save *_target_last.pth.
No JAFFE validation loader, per-epoch evaluation, early stopping, or best-target
selection. Final evaluation uses the last student; EMA teacher is saved too.
All target images participate in unlabeled adaptation and final labeled evaluation:
this is full-pool transductive adaptation, not a verified CAST JAFFE split.
No target truth is used for training, filtering or source checkpoint selection.

jaffe_models contains source/target checkpoint files; jaffe_logs contains automatic
console *.log plus *_ds_diagnostics.csv. Logs use exclusive creation to prevent silent
overwriting: change run_name for another run. Without directory arguments, JAFFE
also defaults to these two directories relative to the current working directory.

At 213 images/batch128, target adaptation remains 2 updates per epoch, 60 total;
source pretraining has its own full source passes. Retaining 30 epochs does not make
optimization exposure equivalent to FER. Thresholds/EMA/JS were not retuned and no
performance improvement is guaranteed solely from preprocessing.

## Checks and limits

python -m unittest discover -s tests -p 'test_jaffe*.py' -v
python -m py_compile train.py dataset.py prepare_jaffe.py training_log.py

Seven tests pass: parsing/duplicates/hidden-label tuples, synthetic similarity geometry,
no-detection errors/face choice, no-checkpoint CLI, logging without overwrite.
The authoring environment lacks torch, OpenCV, TensorFlow and real JAFFE images;
actual detector inference, TIFF decoding and end-to-end training remain untested.
