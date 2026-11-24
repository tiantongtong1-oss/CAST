# Emotion-group radius rescue with source prototypes

Base branch is `experiment/emotion-group-radius`.

The user-defined positive group is surprise/happy/neutral (CAST IDs 0,3,6).
The negative group is fear/disgust/sad/angry (1,2,4,5).

For each class, L2-normalize the source training features, take their mean,
and L2-normalize that mean to obtain the class prototype. Each source sample
contributes its Euclidean distance to the prototype of its own class.
Every ready class has equal total weight within its group, with a weight of
1/n_c per distance. Sort distances in ascending order and take the first
distance whose cumulative weight reaches the requested percentile.

- Positive radius is the class-balanced Q85 of the positive group.
- Negative radius is the class-balanced Q90 of the negative group.

A class needs at least two valid source samples and a nonzero prototype.
Target queries use the source prototype of their predicted class.
Refresh clears all readiness flags. Radius acceptance uses strict distance < R.
Only source training features and source labels determine radii.
Target true labels are used only after selection in the offline audit.

Low-confidence rescue requires two-view agreement,
its class-specific distance gate, both kNN domains predicting the assigned label,
and fused support >= knn_support_threshold (default 0.7).
No separate rescue support threshold.

Training checkpoint names contain `classproto_emotionr_q85q90`.
--checkpoint initializes model weights, not a full training resume.
Existing model/EMA weights are compatible; audit rebuilds banks from source train.
The radius bank stores `class_prototypes`, `group_radii`, and their readiness flags.

Audit retains all per-stage metrics and stages.csv, records grouping and radii in
metadata, accesses validation only for overlap diagnostics, never test.
The inherited training entrypoint still evaluates test once after model selection.

Physical GPU 1: CUDA_VISIBLE_DEVICES=1 python train.py ...
The visible GPU is remapped to logical cuda:0. For audit use the same prefix and --device cuda.

Tests: python -m unittest discover -s tests -p 'test_emotion_radius.py' -v
