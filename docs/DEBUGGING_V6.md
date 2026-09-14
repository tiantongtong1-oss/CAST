# CAST v6 debugging plan (RAF-DB -> FER2013)

This patch is intentionally diagnostic-first. The training code is split into the four blocks in the requested architecture:

1. `cast_v6/model.py` - shared backbone / classifier and target-BN policy.
2. `cast_v6/pseudo.py` - dual-view EMA teacher, CATM thresholds, pseudo-label bank.
3. `cast_v6/ddrl.py` - class-conditional MK-MMD (intra-class alignment + inter-class separation).
4. `cast_v6/ccdr.py` - class-level volume weights and classifier cosine modulation.

`train_v6.py` owns only orchestration, loss composition, logging, checkpointing and health guards.

## Why v5 can collapse even when the block diagram is reasonable

The architecture is not the same thing as a stable implementation. The v5 trace has four strong failure signals:

- The source model starts around 0.49-0.50 on FER2013, but the EMA model drops sharply while the student stays near 0.49. That points to teacher state / BN-buffer handling, not to an intrinsically bad source classifier.
- Pseudo labels fall from thousands to tens. Once this happens, target supervision and DDRL are effectively disabled and the model trains mostly on source replay.
- Predicted FER class distribution collapses into class 3 while classes 1/6 almost disappear.
- `ddrl_min_class_samples=12`, class caps, quantile thresholding, keep-fraction filtering, entropy/margin filters and prototype guards are stacked together. Each filter may be defensible in isolation, but the conjunction is a starvation mechanism.

The paper is materially simpler: class-adaptive thresholding, source+confident-target CE, class-conditional MMD and CCDR. Diagnose that path first, then add guards one at a time.

## Deliberate v6 changes

### EMA / BatchNorm

- Recalibrate **backbone BatchNorm2d only** on target weak views.
- Do **not** recalibrate the 7-dimensional classifier-output BN.
- Freeze BN running statistics during target adaptation.
- Copy teacher buffers exactly from the student after every EMA update; do not EMA-average BN running statistics.
- Apply early-step EMA decay correction, so `0.999` does not make the teacher effectively frozen for the first hundreds of updates.

### Pseudo labels

- Build a pseudo-label bank over the **whole target training set once per epoch** (first forward process).
- Use two weak EMA views and require view agreement.
- Use the paper CATM rule. No quantile threshold, global keep fraction, per-class cap or prototype gate by default.
- Confidence is a weight, not another hidden selection stage.
- Target ground truth is used only when `--debug-target-labels` is passed, and only for logs.

### Classification

- Train the student on source CE plus confidence-weighted target pseudo-label CE on the strong view.
- Do not replace the paper's target classification term with a near-zero KL-only objective.

### DDRL

- `ddrl_min_class_samples=2` by default because MMD only mathematically requires two samples for the unbiased within-domain term. A minimum of 12 caused frequent class deactivation in v5.
- Missing classes are skipped and the remaining valid classes are re-normalized.
- DDRL has the same conservative schedule that previously produced a stable run: 0 for the first 5 epochs, 0.01 for the next 5, then 0.03. Only increase toward the paper's 0.3 after the pseudo pipeline is healthy.

### CCDR

- Class representation density is computed as an actual positive RBF density. The public reference code applies `1 / KernelDensity.score_samples(...)`, but sklearn `score_samples` returns **log density**, not density. That does not implement the paper's `1/rho` volume equation.
- Classifier modulation excludes diagonal constants and logs mean off-diagonal cosine similarity.

## Required ablation ladder

Do not tune the full model first. Run these in order and keep the same source checkpoint and data split.

1. **DATA**: `tools/audit_cast_data.py`. Verify RAF train counts `[1290,281,717,4772,1982,705,2524]`; FER train counts `[3171,4097,436,7215,4830,3995,4965]`; FER private test counts `[416,528,55,879,594,491,626]`. Open sample paths from every class to verify semantics.
2. **SOURCE**: source checkpoint -> FER test, no BN recalibration, no target training.
3. **BN**: source checkpoint + backbone-BN recalibration only. If accuracy jumps and later EMA falls, the issue is state handling, not the dataset.
4. **ST**: EMA + dual-view + CATM + source/target CE, set `w2=0`, `w3=0`. Pseudo selected ratio should stay in the tens of percent, not fall below 1%.
5. **+CSCM**: turn on `w3=0.1` only.
6. **+DDRL**: turn on the 0/0.01/0.03 schedule. Track active classes, intra MMD and inter MMD separately.
7. **FULL CCDR**: enable class-volume weights.
8. Only after 1-7 are stable, test stronger filters (margin, entropy, recovery, prototype gates) **one at a time**.

## Reading the new log

Healthy pseudo block, roughly:

```text
[Epoch 0][Pseudo] agreement=0.72 thresholds=[...] predicted=[...] selected=[...] selected_total=8000+ ratio=0.28 pseudo_acc=0.75+
[Epoch 0][Health] max_pred_ratio=<0.45 pred_entropy=>0.75 selected_classes=7 selected_ratio=>0.15 status=OK
```

Warning patterns:

- `EMA acc << Student acc` immediately after an update: teacher/buffer bug.
- `selected_ratio < 0.02`: pseudo-label starvation; do not tune DDRL yet.
- `selected_classes < 5`: class coverage collapse; inspect CATM thresholds and teacher predicted distribution.
- `max_pred_ratio > 0.55`: head-class collapse.
- `DDRL active <= 2`: alignment signal is too sparse to trust.
- `source CE rises while target CE falls`: catastrophic forgetting or excessive target weight.
- `classifier_cos` increases toward 1: CSCM is not separating classifier vectors.

## Data vs. algorithm decision tree

- Wrong class counts or visibly wrong class samples -> data preprocessing / label map.
- Source RAF accuracy bad -> source training / checkpoint / preprocessing.
- Source RAF good, source->FER bad, but BN-only improves -> domain-statistics problem.
- Source->FER reasonable, EMA alone collapses -> EMA/BN implementation.
- EMA stable, pseudo accuracy good but selected count tiny -> threshold/filtering problem.
- Pseudo labels healthy with `w2=0`, but collapse begins when DDRL turns on -> DDRL scale/MMD/CCDR weighting problem.
- Everything stable but final accuracy remains ~55-58% vs paper ~62% -> compare face alignment, validation/test protocol, backbone checkpoint, augmentation and exact implementation details.

## Protocol warning

The paper describes FER2013 as 28,709 train + 3,589 validation + 3,589 test. If your local tree only contains `train/` and one `test/`, you are missing a validation split. Do not select checkpoints on the test split. The public CAST code does use its target `phase='test'` loader for `best_acc`, so blindly reproducing the released script introduces target-test model-selection leakage.
