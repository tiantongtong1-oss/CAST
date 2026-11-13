# Source-only balanced classifier retraining

Branch base: outside-radius-diagnostics b128968. Existing train.py is unchanged.
Run retrain_balanced_head.py on an existing trusted source checkpoint. No target
path/labels are accepted. All non-fc parameters and buffers are frozen, including
output bn after fc. Backbone is in eval mode; deterministic center-crop raw features
are cached once in CPU RAM, without L2 normalization (matches original fc input).
Only fc is optimized through the fixed output BN using ordinary cross entropy.
Default reset initializes a new fc (cRT-style); --head_init keep is a separate
fine-tuning control. --sampling balanced uses inverse source TRAIN-split frequency
with replacement, N draws per epoch. Expected class balance, not exact batch balance.
--sampling instance is the matching reinitialization/training control.

A stratified 10% source subset is held out from HEAD updates. IMPORTANT: the loaded
source checkpoint may already have seen those images; this is not an independent
representation-level validation set. It is used solely to monitor/select this head
experiment, never a claim of unseen-source generalization. Split paths are recorded.
Best checkpoint selected by source-val Macro-F1; last is also saved. No target-based
selection. Non-fc state digest checked each epoch. Saved model has no ema_teacher,
so scan_candidates.py --weights model/auto loads the retrained head correctly.
Outputs: head_best.pth, head_last.pth, metrics.csv, baseline/best source-val metrics,
split.json. Output directory must not exist; old models cannot be overwritten.

For a meaningful ablation compare the original checkpoint, balanced/reset, and
instance/reset with same seed/split/lr/epochs. Optional balanced/keep isolates the
head initialization effect. Do not conclude representation failure from one failed
head run; lr, frozen BN transform and initialization affect the result. This is a
controlled cRT-style experiment with CAST's fixed output BN, not exact original cRT.

Use a separate worktree, explicit data/checkpoint paths and an idle GPU. After head
training, scan head_best.pth with existing scan_candidates.py. Candidate features,
medoids and source KNN should stay unchanged because only fc changed; teacher
probabilities/class predictions can change. Target labels in scans are evaluation
only. Report per-class precision/recall, UAR/Macro-F1 and overall accuracy, not just
more tail predictions. Summary scan class rows report recall, not class precision.

Checks: Python compilation; split logic checked without torch. Added tensor tests
for frozen feature/BN state, head update, class metrics, stratified split. PyTorch
not installed in authoring environment: tensor tests and full GPU run NOT executed.
On server: python -m unittest discover -s tests -p test_balanced_head.py -v
