# Stable rescue with competing-class evidence (v24)

Base: v22 ad53a5f053d5cbf6694dfcba74de872e9108a59a, not v23.
Branch: experiment/stable-rescue-competition-v24.
No source class weighting or per-class top-k quotas. K=11 selects source neighbors,
not a required number of target pseudo labels per class. No forced backfill when a
class has few trustworthy samples. This avoids forced noisy labels but does not
solve missing tail-class coverage or guarantee precision improvements.

Direct high-confidence dual-view acceptance is unchanged from v22. Low-confidence
agreeing candidates after rescue warmup must satisfy ALL:
- v22 radius OR exact six-of-eleven vote;
- candidate probability minus strongest competitor >= .10 in BOTH weak views;
- distance to closest other ready SOURCE prototype minus distance to candidate
  SOURCE prototype > .02 (strict). Uses existing mean of normalized source features,
  matching the radius bank; no circular dependence on target prototypes;
- candidate source votes minus strongest competing class votes >= 3, full 11
  references available. This is candidate vs competitor, not arbitrary top-1/top-2.
Missing/invalid geometric or neighbor evidence fails closed.
Radius alone cannot bypass competition checks. A 6-versus-5 vote fails; 6/3/2 can
pass. A radius pass with 5/2/2/2 may pass the vote-gap check even without six votes,
provided every other check passes. This preserves an independent radius OR path.
Defaults are starting hyperparameters, not calibrated accuracy guarantees.

Eligibility for stable anchors remains final_accept AND source_radius_pass.
Same label for promotion_epochs consecutive eligible epochs (default 3), no
same-epoch acceleration. A previously promoted sample must remain eligible now.
High-confidence labels immediately supervise strong-view target CE. Rescue labels
supervise CE AND prototype consistency loss only when currently stable anchors.
All target prototype writes use anchors only. Source CE/affinity, EMA, adaptive
thresholds and source prototype initialization remain v22 behavior.
Role separation is required; immediate-prototype and no-role flags are rejected.

Final_Accept_Num / Pseudo_Distribution_Final include pending rescue candidates.
Supervised_Num / Supervised_Distribution report samples actually used in losses.
Pending_Rescue_Num is the unsupervised pending subset. Competition_Pass_Num counts
rescue candidates passing new audits before radius-OR-vote combination.
Existing rescue route counters describe RAW radius/vote routes before competition;
Rescue_Radius_Num and Rescued distributions describe competition-qualified rescue.
CSV accepted means qualified rescue, not immediate learning. Added gap columns.
Prototype_Update_Num must equal Anchor_Num; Immediate_Update_Num must be zero.

## Run from the same source weights as v22

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 0 --epochs 30 --lr 0.001 \
  --promotion_epochs 3 --radius_rescue_warmup_epochs 3 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 --ds_k 11 \
  --rescue_probability_margin 0.10 --rescue_prototype_margin 0.02 \
  --rescue_vote_margin 3 --run_name stable_rescue_competition_v24 \
  2>&1 | tee test_logs/stable_rescue_competition_v24.log
```

Checkpoint loading reads checkpoint['model'] strictly, not optimizer or target
state. Teacher, prototypes and histories are reinitialized. pre_epochs=0 skips
source retraining, not source data usage during adaptation. Run target adaptation
to completion, select by validation, then evaluate test. No need to rerun unchanged
source pretraining for this comparison. A bad source checkpoint still limits
results; the filename alone cannot establish its provenance. Source retraining
is a separate experiment: omit checkpoint and use pre_epochs=30, a new run_name.

Validation: 6 competition/mask tests and 4 v22 vote tests pass, syntax/diff checks
pass. PyTorch/CUDA and RAF/FER data unavailable here; no full training, gradient,
or end-to-end promotion verification. Existing role tests require PyTorch.
