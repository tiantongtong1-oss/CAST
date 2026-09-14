# MobileNetV2 + EMA Teacher + Dual View + Relative Class-Adaptive Threshold

Branch: `improve/ema-dualview-cast`

This branch is a controlled ablation against the stable CAST baseline. It keeps the baseline source stage, MK-MMD DDRL, CCDR class-density weighting, classifier modulation loss, optimizer/scheduler behavior, and strong-view student training. The active target-stage changes are limited to:

1. an EMA teacher initialized from the best source-stage student;
2. two independently augmented weak target views;
3. strict dual-view pseudo-label filtering;
4. a relative class-adaptive threshold that does not collapse every class to 0.9.

No class-distribution correction, temperature scaling, prototype memory, prototype affinity loss, target-loss ramping, or pseudo-label reweighting is active.

## Relative class-adaptive threshold

For each predicted class `c`, the EMA teacher estimates the mean confidence `mu_c` on the full FER2013 training split. Let `mu_bar` be the unweighted mean of the valid class means. The threshold is:

```text
tau_c = clip(tau_0 + beta * (mu_c - mu_bar), tau_min, tau_max)
```

Default values:

```text
tau_0   = 0.85
beta    = 0.50
tau_min = 0.75
tau_max = 0.95
```

Easy/high-confidence classes receive a stricter threshold and difficult/low-confidence classes receive a lower threshold. The previous multiplicative `phi * class_mean * stage_factor` rule is not used because it saturated most classes at 0.9. `--phi` remains accepted only for command-line compatibility with earlier experiments.

## Strict dual-view rule

For target image `x`, draw independent weak views `x_w1`, `x_w2` and strong view `x_s`. The EMA teacher predicts probabilities `p1`, `p2` on the weak views. A pseudo label is accepted only when:

```text
argmax(p1) == argmax(p2) == c
and
max(p1) >= tau_c
and
max(p2) >= tau_c
```

The accepted label supervises the student's strong view. After each successful student optimizer step:

```text
teacher <- EMA(teacher, student)
```

FER2013 training labels are not used for pseudo-label generation or target optimization.

## Logging

Each target epoch prints:

```text
class mean confidence
global mean confidence
class-adaptive thresholds
Agreement_Num
Confident_Num
Pseudo_Distribution
```

This makes it possible to verify that thresholds stay class-dependent and to monitor confirmation bias / class imbalance.

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
  --ema_decay 0.999 \
  --threshold_base 0.85 \
  --threshold_beta 0.5 \
  --threshold_min 0.75 \
  --threshold_max 0.95 \
  2>&1 | tee logs/cast_ema_dualview_adaptive_mobilenet_v2.log
```

Recommended syntax check:

```bash
python -m py_compile train.py dataset.py Networks.py ema_utils.py
```

New checkpoints include `_ema_dualview_adaptive_` in their names, so this experiment does not overwrite either the 54.89% baseline checkpoints or the earlier EMA + Dual View checkpoints.
