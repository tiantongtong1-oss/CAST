# L2-normalized kNN update

Both query and neighbor features are L2-normalized before Euclidean distance. Positive feature rescaling does not affect neighbor ranking (apart from floating point/ties). Invalid/zero vectors are filtered before voting. k=11, strict majority, original labels/probabilities, thresholds, prototype losses and classifier inputs are unchanged. Target self-exclusion by ID is unchanged.

This supersedes earlier raw-kNN descriptions. Target-only branch always disables R. Checkpoint suffix includes l2 to preserve raw-distance baseline models. Added scale-invariance regression tests; tensor/GPU tests require PyTorch.
