# Equal-weight source/target soft kNN fusion, no R

Parent: experiment/target-knn-no-radius at 5eeb72fc8635f228cd71e9f518f59a3dac227232.
One teacher, two separate L2-normalized banks refreshed/frozen each epoch.
Source bank: all valid source train features and true labels. Target bank: all valid target train original-view features and full teacher softmax probabilities; no target ground-truth labels are used. Self is excluded by stable dataset ID. No validation/test memory and no radius.

For each high-confidence agreeing target query, take 11 source and 11 other target neighbors. Both full neighbor sets are required; insufficient neighbors rejects rather than silently changing domain weighting.
ps = mean source one-hot labels, pt = mean target probability vectors, mixed = 0.5*ps + 0.5*pt.
Accept only if mixed.argmax equals original pseudo-label AND mixed[pseudo-label] >= --knn_support_threshold (default 0.7, an experimental configurable value, not calibrated accuracy).
Original pseudo-label is unchanged. Source/target class prototype and loss settings, lambda schedule/fixed override, threshold EMA and classifier inputs retain parent behavior.

Logs include memory size, candidate/accepted counts, mean source/target/fused support and domain argmax disagreement. Checkpoint names contain fusedknn_l2_nor and do not overwrite prior source/target models. Different support-threshold runs still share names and need separate output folders/backups.

Validation: syntax and static integration checks run locally; tensor tests cover soft fusion, ID self exclusion, domain disagreement, insufficient memories, eligibility, normalization and argmax gates. PyTorch/GPU training unavailable in editing environment.
