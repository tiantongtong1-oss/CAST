# Equal-class alignment + low-weight soft rescue

Base: v24 e2b617843b382df11b2f5117548d905262faf323, NOT v25/v27/VICReg.
Branch: experiment/class-balanced-align-soft-rescue-v29.
No v25 neighbor-distance calibration, source resampling/weighting, or VICReg.

## Alignment

Target-stage source/target features are selected by existing stable anchor mask
(source true labels included as before). Compute existing unbiased MK-MMD for
classes with >=2 source AND >=2 target anchor samples in a batch; average over
ACTIVE classes, not all seven, and do not multiply by density weights eta.
No eligible class: graph-connected zero. Sparse classes are skipped, never forced
or oversampled. Active_Class_Batches logs actual participation by class; this
cannot create missing Fear anchors. Source true labels and target stable pseudo
labels partition features. No target ground truth is consumed.

Replace original intra-class MMD, do not add it twice. Existing density-weighted
inter-class separation remains exactly -inter/7 with original w2=.3. New equal-class
intra MMD has SEPARATE --balanced_alignment_weight .03. Equal weighting changes
loss scale vs original density weights: .03 is a conservative experiment starting
coefficient, not calibrated optimum. Unbiased MMD may be negative on finite batches.
Source-pretraining forward path unchanged; balanced_target_alignment is an optional
forward argument, used only by target train.py in this branch.

## Rescue classification

Direct high confidence: unchanged immediate hard CE.
Pending rescue not yet a stable anchor: no target CE/prototype consistency.
Stable rescue: average detached Teacher probabilities from both weak views,
soft cross entropy on strong Student view, --soft_rescue_weight .25.
No sharpening/temperature correction. Fixed Teacher probability distribution is
not guaranteed calibrated/correct. Ordinary soft CE (not KL) includes entropy
constant; student gradients are identical to KL with fixed targets.

Classification numerator = source CE sum + high-confidence target CE sum
+ .25 * stable-rescue soft CE sum.
Denominator = source count + high-confidence count + stable-rescue count,
NOT weighted mass. This keeps .25 a real attenuation and avoids normalizing it
away. Source labels all trusted. Overlapping rescue masks are excluded from hard
confidence branch to prevent duplicate supervision.

Rescued targets do NOT also receive hard pseudo-class prototype consistency loss:
that loss uses confidence_mask only. Original anchor-only target prototype updates
remain, including stable rescued anchors and their stable class label. Alignment
and anchor memory still use stable class assignments; this reduces hard CE damage,
it does not remove all pseudo-label risk. Current acceptance must remain valid.

v24 dual-view rules, radius/6-of-11 + competition, three-epoch promotion, EMA,
source reference refresh and checkpoint selection by validation accuracy remain.
No immediate-prototype or no-role bypass allowed. Legacy route counters predate
competition. Active class batches, soft-rescue distribution and losses are printed.

## Run

```bash
mkdir -p test_logs
set -o pipefail
python -m unittest discover -s tests -p test_balanced_soft_losses.py -v && \
python -u train.py \
  --checkpoint ./models/rafdb_fer/mobilenet_v2_rafdb_fer_source_uniform_control_v27_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 0 --epochs 30 --lr 0.001 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --radius_rescue_warmup_epochs 3 --promotion_epochs 3 --ds_k 11 \
  --rescue_probability_margin 0.10 --rescue_prototype_margin 0.02 \
  --rescue_vote_margin 3 \
  --soft_rescue_weight 0.25 --balanced_alignment_weight 0.03 \
  --run_name class_balanced_align_soft_rescue_v29 \
  2>&1 | tee test_logs/class_balanced_align_soft_rescue_v29.log
```

Use SAME uniform source checkpoint as baseline; no adapted VICReg weights.
No source_class_weight/neighbor_distance/VICReg/immediate flags. See validation
Fear Precision/Recall/F1 plus overall MacroF1/UAR, not Fear count alone. Inherited
best model uses validation accuracy; interpreting only final test cannot isolate
individual effects of these two combined changes. Test once per fixed experiment,
not repeatedly tune on test. Exact resumption is not supported (checkpoint weights
initialize fresh Teacher/history as v24). Run name protects prior outputs.

Validation: 10 NumPy v24 competition/vote tests pass; CLI boundary checks,
syntax/diff checks pass. Five Torch loss/gradient tests provided but skipped here
because PyTorch unavailable. Run them on training host before training. No full
training/accuracy improvement claim; monitoring class participation is essential.
