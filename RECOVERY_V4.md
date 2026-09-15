# Recovery v4: bounded target pseudo-label balancing

This follow-up is based on `recovery_v3_s1314`, whose best target-stage validation accuracy reached 57.06% while the final target test accuracy was 55.84%.

## What the v3 log says

The target train split is much less skewed than the accepted pseudo-label stream. Late in v3, accepted pseudo labels are roughly 2.3k surprise, 0.5k fear, 0.4k disgust, 6.4k happy, 3.5k sad, 1.1k angry and 4.0k neutral. Weak target classes therefore contribute far less target CE than dominant pseudo classes even after labeled-source balancing.

At the same time, prototype agreement is already about 99.9%, so increasing prototype pressure is unlikely to add much independent correction. Student validation loss also grows to about 2 while EMA validation loss stays around 1.7, consistent with an increasingly sharp student.

## v4 changes

Run:

```bash
bash scripts/run_recovery_v4.sh --run_name recovery_v4_s1314
```

The experiment keeps the same source checkpoint, seed, dual-view EMA teacher, adaptive thresholds, source balancing and temporal observation mode. It makes three conservative changes:

1. Add bounded inverse-frequency target pseudo-label CE weights. They are computed only from accepted pseudo labels in each mini-batch, never from FER target ground truth. Default power is 0.20 and the raw class-weight ratio is capped at 1.5x. The observed weighted mean is normalized to one to avoid simply increasing total target loss.
2. Reduce `target_soft_weight` from 0.20 to 0.10. The v3 student already reaches a strong validation peak; v4 keeps some calibration from soft targets but gives the rebalanced hard pseudo-label objective more influence.
3. Reduce `proto_weight` from 0.10 to 0.05 because prototype agreement is saturated in v3. This leaves the source-anchored prototype signal active without letting a nearly self-confirming auxiliary loss compete as strongly with CE.

Environment overrides are available:

```bash
CAST_TARGET_BALANCE_POWER=0.15 \
CAST_TARGET_BALANCE_MAX_RATIO=1.4 \
bash scripts/run_recovery_v4.sh --run_name recovery_v4_s1314_p015
```

## What to compare

Use the same seed/checkpoint and compare v3 vs v4 on:

- final target test accuracy (primary outcome, only after validation-based checkpoint selection);
- best target validation accuracy;
- fear/disgust/angry recall on validation and test;
- accepted pseudo-label distribution;
- Student/EMA validation gap;
- Student validation loss after epoch 15.

A successful v4 run should improve final target test accuracy without relying on target-test labels for any training or selection decision. Because the observed v3 validation-to-test gap is 1.22 percentage points, do not treat a small validation-only gain as sufficient evidence.

## Suggested ablation order

If v4 hurts accuracy, first disable only target pseudo-label balancing with `CAST_TARGET_BALANCE_POWER=0`. If that recovers v3 behavior, try power 0.10-0.15 before changing thresholds. If minority recall improves but overall accuracy falls, lower the maximum ratio from 1.5 to 1.3-1.4. Avoid tuning these settings from target test results; use validation behavior and then confirm the chosen configuration once on test.
