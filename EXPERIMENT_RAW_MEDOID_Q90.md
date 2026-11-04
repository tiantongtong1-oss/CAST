# Original + augmented views, lambda mean thresholds, raw global medoid Q90

Parent: experiment/global-medoid-top7-radius-07-or.

1. Each epoch Teacher scans deterministic original target-training views. For each predicted class c compute the mean maximum softmax confidence, then tau[c] = threshold_lambda * mean[c]. Default lambda=1.0; configurable via --threshold_lambda. No old threshold_base/beta/margin/min/max and no clipping. Classes with no predictions receive infinity, disabling high-confidence admission for that class.
2. Teacher views: deterministic original image preprocessing (resize, center crop, tensor and ImageNet pixel standardization; no random augmentation) + one random weak augmentation. The additional strong view remains the student's training input. Predict from the original view. Both predictions must agree; min(conf_original, conf_augmented) >= tau[c] for high-confidence admission.
3. No two-view feature sum or L2 normalization in the distance path. Source reference memory, medoid, target query, and source KNN use raw backbone features. Target reference is the original-view teacher feature only. Image pixel standardization is retained. The inherited prototype consistency loss still uses its own angular features; those prototypes do not define this distance gate.
4. Select the exact source-sample medoid minimizing total raw Euclidean distance. R is the 90th percentile of all source-to-medoid distances, computed by torch.quantile(..., 0.90, interpolation='linear'). Seven classes share this single R. Accept raw target distance strictly < 0.7 R. Rebuild medoid, R, and true-label source neighbor bank each epoch.
5. OR selection unchanged: agreement AND (high confidence OR radius rescue OR source-neighbor rescue). Low-confidence rescue starts at zero-based epoch 3 by default. Outside-radius candidates use nearest 11 true-label source neighbors, strict majority matching the original teacher pseudo-label, minimum 3 available neighbors.

Log prefix: logs/raw_medoid_q90_TIMESTAMP_PID.log. Checkpoint suffix: _raw_medoid_q90_07_or_.

Tests: python -m unittest discover -s tests -p test_raw_medoid_quantile.py -v
Authoring checks: Python syntax, static dataflow/integration, lambda threshold arithmetic checked. PyTorch is unavailable in this environment; tensor tests and full GPU training have not executed.
