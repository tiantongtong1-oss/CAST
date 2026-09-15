# Recovery v3: balanced source + temporal soft targets

This follow-up is based on the completed `recovery_base_s1314` run. The base run reached 56.90% best target validation accuracy but only 55.92% final target test accuracy, below the earlier 56.39% prototype-consistency result.

## Why change the recovery configuration

The completed run shows two consistent failure modes.

1. The accepted pseudo labels remain strongly skewed. Late in training, `happy` has more than six thousand accepted pseudo labels while `fear` remains around five hundred and `angry` around one thousand. Validation/test recall for fear stays near single digits and angry remains much lower than the strong classes.
2. Student validation loss rises above 2 while validation accuracy improves only slowly. This is consistent with increasingly confident fitting of hard pseudo labels: more predictions become confident, but wrong predictions become expensive as well.

Hard temporal filtering is not the default fix here. The previous temporal experiment already showed that a strict streak/history gate can remove useful low-confidence samples and does not directly solve minority-class starvation.

## v3 configuration

Run:

```bash
bash scripts/run_recovery_v3.sh --run_name recovery_v3_s1314
```

The v3 wrapper reuses the same source checkpoint and the same reproducible target-stage setup as recovery v2, then changes only these optimization controls:

```text
temporal_mode = observe
target_soft_weight = 0.20
source_balance_power = 0.50
source_balance_max_ratio = 2.0
```

Everything else remains inherited from `scripts/run_temporal_consistency.sh`, including the MobileNetV2 backbone, dual-view EMA teacher, adaptive thresholds, prototype loss, learning-rate schedule, seed, and 30 target epochs.

### Bounded source balancing

Source labels are clean, so v3 uses the existing target-stage source CE weighting instead of reweighting target pseudo labels. With the RAF-DB counts in this experiment the minority classes are boosted, but the class-weight ratio is capped at 2x. This is intended to preserve source semantic evidence for fear/disgust/angry while the target pseudo-label stream is biased toward happy/neutral/sad.

### 20% temporal soft CE

The temporal bank already stores an EMA of the two weak-view teacher probabilities. In `observe` mode it does not remove any dual-view candidate. Setting `target_soft_weight=0.20` therefore changes target CE from pure hard pseudo-label CE to:

```text
0.80 * hard CE + 0.20 * temporal soft CE
```

This is deliberately much weaker than the earlier 0.5 soft-label experiment. The goal is calibration/regularization, not replacement of the hard pseudo label.

## What to compare

Compare `recovery_v3_s1314` with `recovery_base_s1314` using the same checkpoint SHA, target path-order SHA, seed and worker count. The main quantities are:

- best target validation accuracy;
- final target test accuracy (report only after the validation-selected checkpoint is fixed);
- fear/disgust/angry recall;
- student validation loss trend;
- accepted pseudo-label distribution.

A useful v3 outcome is not merely a higher macro recall. The intended target is to recover the previous 56.39% test level and preferably move beyond it without sacrificing overall validation accuracy.

## If v3 still plateaus

Do not immediately enable hard temporal filtering. The next controlled changes should be tested one at a time: lower `target_soft_weight` to 0.10 if total accuracy falls while minority recall rises; raise it to 0.30 if validation loss still diverges; or reduce `source_balance_power` to 0.35 if happy/neutral accuracy drops too much. Re-run promising settings with additional seeds before treating a sub-percent difference as stable.

This repository change does not claim a measured accuracy improvement because the GitHub execution environment does not contain the RAF-DB/FER2013 data or the GPU training setup. It provides the next configuration selected from the completed run's failure pattern rather than from target test-label tuning.
