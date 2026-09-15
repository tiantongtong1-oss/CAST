# Experiment: Prototype Consistency v1

Base branch: `improve/ema-dualview-cast`

Experiment branch: `experiment/prototype-consistency-v1`

This experiment keeps the best-known EMA + dual-view + stable class-adaptive-threshold pipeline unchanged and adds one representation-level regularizer: source-anchored prototype consistency.

## Motivation

The best branch already filters pseudo labels with two weak views and a stable EMA teacher, but accepted pseudo labels remain class-imbalanced and are selected only from probability confidence. A high-confidence prediction can still be geometrically inconsistent with the class representation.

Prototype Consistency v1 adds a second reliability axis in feature space without using FER2013 training labels.

## Method

1. Build one fixed source prototype per expression class from deterministic RAF-DB train features after the best source checkpoint is restored.
2. Maintain one target EMA prototype per class using only samples already accepted by the existing dual-view confidence rule.
3. Blend source and target prototypes:

```text
P_c = normalize(a * P_source,c + (1-a) * P_target,c)
```

with default `a = 0.5`.
4. For the student's strong target features, compute cosine logits against all blended prototypes and apply a prototype classification loss using the pseudo label.
5. Average the prototype loss per class first, then across classes. This prevents high-volume pseudo classes from dominating the prototype regularizer.
6. Warm up the prototype memory for 3 target epochs with zero prototype gradient, then linearly ramp the prototype loss over 5 epochs.

The existing pseudo-label mask is NOT made stricter in v1. This is deliberate: the experiment isolates feature-space regularization and avoids starving already rare pseudo classes.

## Defaults

```text
proto_weight          = 0.10
proto_temperature     = 0.20
proto_momentum        = 0.99
proto_source_anchor   = 0.50
proto_warmup_epochs   = 3
proto_ramp_epochs     = 5
```

## Diagnostics

Each target epoch additionally logs:

```text
Prototype_Agreement
Mean_Assigned_Cosine
Target_Prototype_Counts
Prototype Loss
ProtoWeight
```

`Prototype_Agreement` measures how often the nearest blended prototype agrees with the already accepted pseudo label. It is diagnostic only in v1 and does not filter pseudo labels.

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
  --threshold_margin 0.02 \
  --threshold_min 0.80 \
  --threshold_max 0.95 \
  --proto_weight 0.10 \
  --proto_temperature 0.20 \
  --proto_momentum 0.99 \
  --proto_source_anchor 0.50 \
  --proto_warmup_epochs 3 \
  --proto_ramp_epochs 5 \
  2>&1 | tee logs/cast_prototype_consistency_v1_mobilenet_v2.log
```

## Interpretation

Primary comparison: best branch test accuracy `56.17%` versus this experiment using the same backbone, source/target split, seed, optimizer, scheduler, and threshold configuration.

Useful signals before the final test result:

- validation accuracy should not collapse when `ProtoWeight` becomes non-zero;
- `Prototype_Agreement` should trend upward or stay stable;
- minority pseudo classes should not disappear from `Target_Prototype_Counts`;
- prototype loss should remain finite and generally decline after the ramp begins.

If v1 improves accuracy, the next clean ablation is prototype-gated pseudo-label filtering. If v1 hurts accuracy, first reduce `--proto_weight` to `0.05` before adding any new mechanism.
