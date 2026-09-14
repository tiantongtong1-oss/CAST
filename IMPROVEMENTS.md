# MobileNetV2 + EMA Teacher + Dual View

Branch: `improve/ema-dualview-cast`

This branch is intentionally reduced to a clean ablation against the stable CAST baseline. It keeps the baseline CAST losses and training schedule, and adds only:

1. an EMA teacher copied from the best source-stage student;
2. two independently augmented weak target views;
3. pseudo-label acceptance only when both teacher views agree and their averaged confidence passes the original CAST class-adaptive threshold.

The following experimental mechanisms from earlier commits are removed from the active training path:

- class-distribution correction / uniform-prior alignment;
- temperature scaling;
- prototype memory and prototype affinity loss;
- target-loss ramping and pseudo-label reweighting;
- additional target-stage optimizer changes;
- custom threshold floors/ceilings beyond the original CAST threshold rule.

The source stage, MK-MMD DDRL loss, CCDR class-density weighting, classifier modulation loss, optimizer, learning-rate scheduler, and source/target loss normalization are kept aligned with the stable `sep09-version` baseline.

MobileNetV2 is the default backbone on this branch.

## Target-stage pseudo label rule

For target image `x`, draw two independent weak augmentations `x_w1` and `x_w2` and one strong augmentation `x_s`.

The frozen-gradient EMA teacher predicts both weak views. Let `p1`, `p2` be their softmax probabilities. A pseudo label is accepted only when:

```text
argmax(p1) == argmax(p2) == argmax((p1+p2)/2)
and
max((p1+p2)/2) >= CAST_class_adaptive_threshold[class]
```

The accepted pseudo label supervises the student's strong view. After each successful student optimizer step:

```text
teacher <- EMA(teacher, student)
```

No FER2013 training labels are used for pseudo-label generation or optimization.

## Run

```bash
python train.py \
  --backbone mobilenet_v2 \
  --pre_epochs 30 \
  --epochs 30 \
  --lr 0.001 \
  --w1 4 \
  --w2 0.3 \
  --w3 0.1 \
  --phi 1.4 \
  --ema_decay 0.999 \
  2>&1 | tee logs/cast_ema_dualview_only_mobilenet_v2.log
```

Recommended first check:

```bash
python -m py_compile train.py dataset.py Networks.py ema_utils.py
python train.py --backbone mobilenet_v2 --pre_epochs 2 --epochs 2
```

The checkpoints include `_ema_dualview_` in their names, so this experiment does not overwrite the saved baseline checkpoints.
