# Distance-filtered adaptive voting (v25)

Base: v24 e2b617843b382df11b2f5117548d905262faf323.
Branch: experiment/adaptive-distance-vote-v25.

At every source evidence refresh (each target epoch), fit per-class distance
thresholds using ONLY current Teacher source features and true source labels.
For each source class, exclude each query itself and record distance to its
min(11, class_count-1)-th same-class neighbor. Take the 90th percentile, controlled
by --neighbor_distance_quantile. Singleton/absent classes cannot calibrate and
fail closed. Features are L2 normalized; distances are Euclidean, not squared.
Zero threshold permits only exactly coincident features. Calibration uses all
source points, block size 64; it adds per-class quadratic distance work each epoch.
No target labels or validation/test samples calibrate these thresholds.

For a target candidate, take at most 11 GLOBAL nearest source references. Filter
by the predicted candidate class threshold (distance <= threshold). Do not replace
removed neighbors with farther ones. Remaining count is K_eff:
0-2: unavailable, no vote rescue.
3-5: all must have candidate label.
6-11: candidate fraction >=75% AND candidate votes minus strongest competitor >=3.
Counts, votes and ratios refer to the distance-filtered neighborhood; no weighted
votes, no per-class quotas and no target backfill. Three close unanimous neighbors
can pass; three far unanimous neighbors cannot.

Preserve v24 probability margin >=.10 in BOTH weak views, source prototype distance
advantage >.02, stable three-epoch same-label promotion, no immediate prototype
updates, and no source class weighting. High-confidence direct acceptance bypasses
neighbor audits. Rescue candidates must pass adaptive vote AND competing-class
audits; radius cannot bypass failed/unavailable adaptive voting. This retains
v24 common-vote-veto semantics, rather than restoring independent radius rescue.
The former (radius OR vote) with mandatory vote simplifies to vote. Radius remains
required for anchor eligibility. Rescue-only samples outside radius cannot become
anchors or supervise until eligible; audit acceptance is not supervised count.
Only stable rescue anchors supervise CE/prototype consistency; only anchors write
prototypes. No change to direct high-confidence supervision.

Logs: Neighbor_Distance_Thresholds; Effective_Neighbor_Count_Histogram_0_to_11;
Supervised_Num; Pending_Rescue_Num; Supervised_Distribution. CSV adds raw neighbor
count, threshold, filtered mean/max distances. source_count is K_eff, vote_ratio
uses K_eff (0 if empty). Legacy raw route counters remain before competition and
must not be interpreted as accepted rescue. DS_Passed is adaptive vote pass.

## Run

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
  --neighbor_distance_quantile 0.90 \
  --rescue_probability_margin 0.10 --rescue_prototype_margin 0.02 \
  --rescue_vote_margin 3 --run_name adaptive_distance_vote_v25 \
  2>&1 | tee test_logs/adaptive_distance_vote_v25.log
```

Same source checkpoint supports controlled comparison; weights initialization,
not resume. Do not pass immediate_prototype_update or source_class_weight flags.
Thresholds are experimental; filtering may reduce tail-class coverage. No claimed
accuracy improvement. Fifteen NumPy tests pass, Python syntax/diff checks pass.
PyTorch/CUDA/RAF/FER unavailable here; full training and gradient integration unrun.
