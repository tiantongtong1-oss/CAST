# v8: v7 full-source flow + source multi-prototype margin

Branch v8, based on experiment/source-multi-prototype-margin-C.
train.py and train_source_c.py now share the SAME integrated training entry point.
Unlike the older 80/20 C script, v8 trains on all RAFDB train (12271 examples for
this dataset), builds K=3 source prototypes from all train fixed views every epoch,
and selects source best by FER validation accuracy. No RAFDB validation split.
Defaults source_margin=0.2, source_margin_weight=0.1, clustering iterations=20.
Source loss: original v7 weighted CE/affinity/classifier modulation plus weighted
multi-prototype hinge. No change to target losses, masks or target prototype bank.

Training batch=128 (v7), same backbone initialization, Adam lr=0.001 weight decay
1e-4, scheduler gamma=.95, transforms and seed. Source prototype refresh restores
RNG state so its extra loader pass does not shift training augmentation RNG.
Original flow reloads source-best model/optimizer/scheduler before target training;
v8 retains that behavior (not a fresh target optimizer). FER validation labels
are used for checkpoint selection, not training gradients or source clustering.
FER test is evaluated only after target checkpoint selection.

## End-to-end C: 30 source + 30 target epochs

```bash
python -u train.py --pre_epochs 30 --epochs 30 --source_prototypes_per_class 3 --source_margin 0.2 --source_margin_weight 0.1 --knn_bandwidth_multiplier 22.627417 --run_name v8_C_full > v8_C_full.log 2>&1
```
Do not provide an old checkpoint for the from-scratch pretrained-backbone comparison.
train_source_c.py accepts these identical arguments. Old --margin/--margin_weight
arguments belonged to the previous split script; v8 uses --source_margin and
--source_margin_weight. Do not use an existing run_name to avoid model overwrites.

## Matched A (same entire flow; new loss disabled)

```bash
python -u train.py --pre_epochs 30 --epochs 30 --source_margin_weight 0 --knn_bandwidth_multiplier 22.627417 --run_name v8_A_full > v8_A_full.log 2>&1
```

Models: new_models/rafdb_fer/v8_C_full/
mobilenet_v2_rafdb_fer_source_global_gaussian_knn_v7_source_best.pth
mobilenet_v2_rafdb_fer_source_global_gaussian_knn_v7_target_best.pth
Names retain v7 compatibility; v8 identity/configuration is recorded in args and
isolated run directory. Target selection remains FER validation accuracy.
Use --source_only for source training and selection only (requires pre_epochs>0).
Do not describe FER-validation-based selection as a fully target-label-free protocol.

Verification: syntax checks and CPU tests, including flow/selection/RNG assertions.
PyTorch gradient checks are supplied but skipped locally because torch/CUDA is not
installed. Full GPU training requires server validation. No performance guarantee.
