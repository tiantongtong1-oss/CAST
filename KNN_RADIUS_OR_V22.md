# Radius OR exact KNN majority rescue

Base: v19 commit 12c76107d6146d5d7763550674e5f9252934f530.
Branch: experiment/radius-or-knn-vote6of11-v22.

Keep the seven original class-adaptive confidence thresholds, dual-view agreement,
EMA and source radius quantile. Confidence accepts bypass rescue.
For low-confidence agreeing targets, after the three-epoch rescue warmup:
rescue = radius_pass OR vote_pass.
Radius uses strict normalized feature-to-source-mean distance < class radius.
Vote uses exact 11 nearest labeled SOURCE references by squared Euclidean distance
between L2-normalized features. Distance ordering equals cosine ordering, but votes
are unweighted. Vote passes iff at least 6 of 11 true source labels equal Teacher's
pseudo-label (ratio >=6/11). Fewer than 11 references means vote unavailable;
radius may still rescue independently. Zero/nonfinite query features fail voting.
No label correction: a failed rescue remains uncertain and excluded from target supervision.

No Otsu scan, no .92 threshold, no DS fusion. Prior mean-support CLI options are removed;
do not pass --support_threshold_mode or --support_fixed_threshold. Legacy DS CLI values
remain compatibility-only; ds_k is constrained to 11. ds_min_neighbors does not relax
full-neighborhood requirement. Source-only references; query IDs have separate namespace.
Role eligibility stays final_accept AND radius_pass; same-label 3-epoch promotion unchanged.
Immediate prototype update stays optional and is enabled in the proposed run command.
Classification loss weights, prototype loss and affinity are unchanged. No class balancing,
no resolution/grayscale augmentation. The inherited source-only statistics crash is fixed.

CSV retains old columns for compatibility; support/mass now mean vote fractions,
unknown is zero for available neighborhoods, conflict is not computed. Added vote count,
ratio, radius pass and vote pass. DS_Passed counts vote passes, not the OR union;
Radius_Rescue_Num counts the OR union. Route counters are mutually exclusive.
Removing isolated calibration does not consume training RNG, but training trajectories
can differ as masks change. Existing old diagnostic scripts/tests refer to older rules;
use tests/test_neighbor_vote.py for this branch's vote behavior.

Validation: four NumPy tests pass (6/11, 5/11, distance ranking, invalid/short bank, OR);
Python syntax passes. Full PyTorch/CUDA/RAF/FER training not available. No accuracy claim.
