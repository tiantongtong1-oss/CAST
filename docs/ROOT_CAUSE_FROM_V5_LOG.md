# Root-cause readout from the supplied v5 trace

The failure timing is unusually informative.

## 1. DDRL is not the cause of the early collapse

`applied_w2` is `0.0000` for epochs 0-5 because `affinity_warmup=6`. The class-distribution / pseudo-label collapse is already obvious at epochs 2-3. Therefore debug the EMA/pseudo/BN/backbone path before touching the DDRL math.

## 2. Epoch 2 matches the backbone-unfreeze boundary

The config has `freeze_backbone_epochs=2`.

- epoch 1: student 0.4826, EMA 0.4745
- epoch 2: student 0.4962, EMA 0.2914

That is exactly when the backbone begins to move while an EMA teacher with `decay=0.999` can remain stale. A hard frozen->unfrozen transition is especially dangerous if teacher BN buffers are not copied exactly.

v6 removes the hard transition and instead trains the backbone from the beginning with a smaller LR (`target_lr * backbone_lr_mult`).

## 3. Epoch 3 matches the anchor-guard expiry

The config has `anchor_guard_epochs=3`.

At epoch 3:

- `anchor_agree` drops from about 0.606 to 0.348;
- `trusted` collapses to roughly `[49,17,0,15,0,0,0]`;
- `selected` becomes only 24 samples;
- all seven classes report `need_recovery=1`;
- predicted classes become highly distorted.

The sequence is consistent with: backbone transition -> stale/bad EMA teacher -> anchor guard removed -> pseudo selector follows the bad teacher -> starvation / head-class feedback loop.

## 4. The selector has too many conjunctive gates

The v5 config simultaneously uses CATM/quantiles, a keep fraction, entropy, margin, temporal streaks, prototype margin, anchor confidence, class caps and starvation/recovery logic. Each condition reduces recall. Once one class becomes scarce, the class cap and temporal requirements can make it mathematically difficult to recover.

The supplied earlier run selected 8,517 confident samples at epoch 0 and 18,794 by epoch 29 while reaching a reported best target accuracy of 0.5765. v5 selects 0 / 2,434 / 1,809 / 24 / 32 / 65 in epochs 0-5. That is a selector-regression signal, not a subtle hyperparameter difference.

## 5. Recovery is not functioning as a recovery mechanism

v5 repeatedly prints `need_recovery=1` while `recovery=[0,...,0]`. Check the semantics of `recovery_topk_per_class=0`; if zero means select zero rather than auto/unlimited, recovery is effectively disabled. Regardless, a recovery stage should not be the primary fix: restore a healthy teacher and selector first.

## 6. Source data is not the first suspect

The RAF-DB and FER2013 class counts in the current run match the counts in the earlier, better run. The FER vector `[3171,4097,436,7215,4830,3995,4965]` is also the expected FER2013 training distribution after reordering to CAST's class order `[surprise,fear,disgust,happy,sad,angry,neutral]`.

Data still needs an audit (semantic folder mapping, preprocessing, missing validation split, face alignment), but the abrupt epoch-2/3 phase transition is an algorithm/state-management signature.
