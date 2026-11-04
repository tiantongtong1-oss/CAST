# Global medoid radius OR rescue

Based on experiment/class-prototype-top7-radius-07. All seven classes share ONE reference point and ONE radius.

Each epoch, extract deterministic L2-normalized source features with the current EMA teacher. Pick an actual valid source sample minimizing the sum of Euclidean (not squared Euclidean) distances to ALL valid source samples, without per-class balancing. Ties choose the earliest memory row. Compute R as the mean of the seven largest Euclidean distances to this medoid. Target radius acceptance is strictly distance < 0.7 R, independent of predicted class. Fewer than seven valid source samples disables radius rescue; zero/nonfinite features fail closed.

The initial R is computed at epoch 0 and refreshed, together with the medoid, at each subsequent epoch, following the parent OR experiment. Source pair distances use float64, exhaustive chunked evaluation with O(chunk*N) temporary memory. This is exact exhaustive medoid selection up to floating-point numerical precision, not sampling or nearest-to-mean approximation. Computation remains quadratic in source sample count and may be slow on consumer GPUs.

Final acceptance: dual-view agreement AND (high confidence OR global-radius rescue OR source-neighbor rescue). Default rescue starts at zero-based epoch 3. Outside-radius low-confidence candidates use nearest 11 source samples with strict majority matching the unchanged teacher pseudo label; at least 3 available neighbors required. Class prototypes/target EMA prototypes remain only for the inherited prototype consistency loss; they do not define the global-radius gate. True source labels are used for neighbor votes. Target training true labels are never used.

Logs: logs/global_medoid_TIMESTAMP_PID.log. Checkpoints use _global_medoid_top7_07_or_ suffix. Diagnostics print Global_R, Cutoff_07R, Medoid_Memory_Row, readiness, and separate radius/neighbor rescue counts.

Tests: python -m unittest discover -s tests -p test_global_medoid_radius.py -v
Syntax/static integration checks passed in authoring environment. PyTorch is unavailable here; tensor numerical tests and full GPU training have not run. The previous experiment documents describe their respective historical variants.
