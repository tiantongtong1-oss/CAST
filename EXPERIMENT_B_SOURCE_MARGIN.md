# Experiment B — source single-prototype margin

Branch: experiment/source-single-prototype-margin-B
This implements B only: one detached normalized prototype per source true class,
and L_margin=mean relu(margin + hardest_other_cosine - true_class_cosine).
L_total = w1*CE + w2*CAST affinity + w3*classifier modulation + margin_weight*L_margin.
Defaults: margin 0.1, margin_weight 0.1. These are starting settings, not validated
optimal values. No multi-prototype mechanism or new target acceptance gate.

## Clean comparison

Run from CAST root with CUDA/PyTorch installed:

```bash
python -u train_source_b.py --epochs 30 --margin_weight 0 --run_name source_A > source_A.log 2>&1
python -u train_source_b.py --epochs 30 --margin_weight 0.1 --margin 0.1 --run_name source_B > source_B.log 2>&1
```
No checkpoint argument: both initialize the existing pretrained backbone identically.
RAFDB train is stratified 80/20 using split_seed=2000. Train augmentation applies
only to train IDs. Validation and prototype views use fixed transforms. Prototypes
refresh once per epoch from training IDs only; validation/test never enter prototypes.
Each validation pass prints all class precision/recall/macro-F1; class F1 can also
be computed from reported precision/recall. Selection uses only RAFDB validation
macro-F1, with RAFDB test evaluated once after selection. FER is not used here.
A/B share split, seed and other training settings. No full resume semantics.

Outputs: new_models/rafdb_fer/source_B/source_B_best_macro_f1.pth and split.json.
A uses its own run_name directory (same checkpoint filename). Existing output
folders are refused to prevent overwriting. All checkpoints from train.py on this
branch also go under new_models/rafdb_fer/<run_name>.

## Fast diagnostic fine-tuning (not a clean paper comparison)

```bash
python -u train_source_b.py --epochs 10 --lr 0.0001 --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_source_global_gaussian_knn_v7_source_best.pth --run_name source_B_finetune > source_B_finetune.log 2>&1
```
The old checkpoint has seen the newly held-out validation examples; a warning is
printed and recorded in split.json. Use clean initialization for publication claims.

## Audit selected B on FER, without target training

```bash
python -u train.py --pre_epochs 0 --checkpoint new_models/rafdb_fer/source_B/source_B_best_macro_f1.pth --audit_only --run_name audit_source_B
```
Compare source validation/test and FER disgust F1, macro-F1, precision/recall.
An improvement is not guaranteed. Repeated seeds are required for stable claims.

## Checks

python -m unittest discover -s tests
python -m py_compile train_source_b.py source_margin.py train.py
The editing environment lacks PyTorch/CUDA: margin gradient test is provided but
skipped there; run it on the training server before GPU training.
