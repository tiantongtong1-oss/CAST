# Emotion-group radius rescue

Base: experiment/class-medoids-strict-dual-domain-rescue.

Seven-class prediction labels and per-class medoids remain unchanged.
The user-defined positive group is surprise/happy/neutral (CAST IDs 0,3,6).
The negative group is fear/disgust/sad/angry (1,2,4,5).

Each source training sample contributes its normalized Euclidean distance to its OWN
class medoid. Each ready class has equal total weight in its group (sample weight 1/n_c).
Use weighted empirical inverse CDF without interpolation:
- Positive radius: group Q90.
- Negative radius: max(original all-class balanced Q90, negative group Q95).

The negative floor guarantees that negative rescue is no tighter than the old shared
radius. It does not guarantee negative radius numerically exceeds positive radius.
A class needs at least two valid source samples; missing classes cannot rescue.
Refresh clears all readiness flags. Radius acceptance uses strict distance < R.
Only source training features and source labels determine radii.
Target true labels are used only after selection in the offline audit.

Keep confidence path unchanged. Low-confidence rescue requires two-view agreement,
its class-specific distance gate, both kNN domains predicting the assigned label,
and fused support >= knn_support_threshold (default 0.7).
No separate rescue support threshold.

Training output names now contain classmedoid_emotionr instead of classmedoid_sharedr.
--checkpoint initializes model weights, not a full training resume.
Existing model/EMA weights are compatible; audit rebuilds banks from source train.
New bank state includes group_radii, group_radius_ready, class_radius_group.

Audit retains all per-stage metrics and stages.csv, records grouping and radii in
metadata, accesses validation only for overlap diagnostics, never test.
The inherited training entrypoint still evaluates test once after model selection.

Physical GPU 1: CUDA_VISIBLE_DEVICES=1 python train.py ...
The visible GPU is remapped to logical cuda:0. For audit use the same prefix and --device cuda.

Tests: python -m unittest discover -s tests -p 'test_emotion_radius.py' -v
