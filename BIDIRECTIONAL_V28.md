# Bidirectional RAF-DB / FER2013 adaptation (v28)

Based on experiment/global-mean-std-cast-ce-v27-run-history, commit
0746864fcf43ebdc4a285da6eb3f336154283dbf. Both directions use the same
global mean/std/epoch-EMA threshold, rescue competition, stable anchor promotion,
and CAST-equivalent classification loss (all source+target examples in the
denominator). Per-run logs/checkpoints/metadata archives are retained.

Select direction with --data1 (source) and --data2 (target), and supply matching
--source_path and --target_path. Paths are not swapped automatically.
FER labels remain mapped into the common RAF/CAST class order.

When RAF is the target, --raf_val_fraction 0.10 and --raf_split_seed 2000 create
a reproducible stratified validation holdout from official TRAIN. Holdout images
are excluded from adaptation and threshold scans. Official TEST is used only for
final evaluation. This uses labeled target validation for checkpoint selection,
as in the base implementation; it is not a target-label-free selection protocol.
Training/threshold views have identical ordering and stable local sample IDs.
RAF as source still uses its complete official training split.

The base branch already fixes the undefined target_weight pretraining log.
No complete GPU training was run.

## FER2013 -> RAF-DB, with source pretraining

```bash
python -u train.py \
  --data1 fer --data2 rafdb \
  --source_path /workspace/ttt/code/data/fer2013 \
  --target_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --backbone mobilenet_v2 --pre_epochs 30 --epochs 30 \
  --lr 0.0001 --workers 0 \
  --raf_val_fraction 0.10 --raf_split_seed 2000 \
  --threshold_std_scale 0.5 --threshold_smoothing 0.8 \
  --threshold_min 0.85 --threshold_max 0.99 \
  --run_name fer_to_raf_cast_ce_history_v28
```

With a compatible FER source checkpoint, use --checkpoint /path/to/fer_source.pth
and --pre_epochs 0 instead. Checkpoints must contain a matching checkpoint['model'];
this initializes weights, not exact training resume. Do not describe RAF-pretrained
weights as FER source-only weights. Source pretraining checkpoint selection uses
the held-out target validation set, matching the base training procedure.

For RAF-DB -> FER2013 swap data1/data2 and paths, and choose a distinct run_name.
Default learning rate remains 0.001; the example explicitly selects 0.0001.

## Validation

```bash
python -m py_compile train.py dataset.py
python -m unittest discover -s tests -p test_bidirectional_data.py -v
python -m unittest discover -s tests -p test_global_mean_std_ema.py -v
```

Five bidirectional tests exercise actual dataset bodies with stubbed image/Torch
dependencies (real NumPy/Pandas), disjoint holdouts and stable IDs, view contracts,
FER class mapping, both routing directions, parser validation and the pretraining
regression. Six existing scanner tests pass. PyTorch/CUDA/OpenCV and real datasets
were unavailable; full forward/backward training remains unverified.

Run archives remain under --history_root (default ./model_history); each invocation
prints Run_History and saves logs/train.log, diagnostics, models and run.json.
Additional checks: test_cast_classification.py and test_run_history.py.
