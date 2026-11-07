# Soft classification weighting outside shared Q90 radius

Parent: experiment/class-medoids-shared-radius at fe51e4efd860a9358033396de77db8ba547e1675.
Candidate selection is unchanged: agreement AND (confidence OR radius fallback) AND fused kNN. High-confidence samples outside R remain eligible.

After radius warmup, final accepted high-confidence targets outside R receive --outside_radius_cls_weight (default .5) on CLASSIFICATION cross entropy only. Source weights stay 1; in-radius targets and radius rescues stay 1. Before warmup all weights are 1. Unaccepted targets retain binary zero mask. con_idx/train_con_idx remain binary for network affinity, prototype consistency and target prototype updates.
Denominator remains sum of binary train_con_idx, preserving source classification coefficients. This deliberately lowers total target classification contribution; it is not normalized by sum of soft weights. Weight=1 recovers baseline loss behavior.
Strong Student view, two-view-mean threshold, shared class-balanced Q90, prototype weight and fusion rules unchanged. Model suffix softweight prevents baseline overwrites, but repeated softweight runs share names.
Logs show outside-R accepted high-confidence count and configured weight.
Validation: syntax plus stdlib production-expression truth-table and loss coefficient tests. Tensor/GPU training unavailable locally.
