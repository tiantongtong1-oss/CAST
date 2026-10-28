# Offline v27 pseudo-label quality

Run `evaluate_pseudo_labels.py` on the full FER2013 training split. This is a
frozen-checkpoint diagnostic with no optimizer, training, backward pass or
checkpoint writes. It reuses v27's dual-view selector, source-radius fitting,
source-only K=11 majority vote and competing-class audit. No model downloads
are required (`pretrained=False`). Target labels are used only for reporting.

```bash
python -u evaluate_pseudo_labels.py \
  --checkpoint ./models/rafdb_fer/mobilenet_v2_rafdb_fer_source_uniform_control_v27_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --weights model \
  --batch_size 128 --workers 0 --seed 1314 \
  --threshold_std_scale 0.5 --threshold_smoothing 0.8 \
  --threshold_min 0.85 --threshold_max 0.99 \
  --rescue_mode enabled \
  --output_dir ./pseudo_eval/source_best_v27
```

Use a new empty output directory each time. All training splits must follow the
existing Dataset conventions (FER native numeric class directories). This
command assesses source initialization, not the outcome of target training.
For a target-best checkpoint change --checkpoint and use --weights auto
(prefers ema_teacher) or explicitly --weights ema_teacher/model. Missing
requested weights fail rather than silently substituting another model.

Threshold, source quantile and competition defaults come from checkpoint args
when present, then v27 defaults; explicit CLI flags override them. The effective
settings and chosen weights are written in summary.json. For controlled
comparisons explicitly use identical options. --rescue_mode enabled evaluates
post-warmup rescue irrespective of checkpoint epoch. Use disabled for the
confidence-only warmup policy.

Outputs:
- pseudo_label_metrics.csv: confidence, rescued and final; seven predicted
  classes plus ALL. accepted_count, correct_count and pseudo_accuracy (0..1).
  Empty selections have blank accuracy, not zero. ALL uses pooled counts.
- samples.csv: each target ID/path, pseudo/true label, correctness, minimum
  confidence, threshold, agreement and the three acceptance flags.
- summary.json: checkpoint, options, effective settings, sample counts,
  thresholds, all class metrics and snapshot limitations.

The final set is confidence OR rescued. It is NOT the actual supervised set:
rescued samples must be promoted by cross-epoch history before they learn.
The script does not advance saved roles, does not fabricate empty-anchor
metrics, and does not reconstruct historical training supervision or anchors.

v27 checkpoints do not save the previous global threshold. By default the
script recomputes mean/std/clipping without EMA history (the existing first
scan rule). To use a known historical threshold pass --threshold 0.86; to
apply the existing EMA formula with a known prior value pass --previous_tau
0.86 instead. These options are mutually exclusive. Fresh augmentations,
refitted source references and a frozen model differ from evolving training
batches. Compare snapshots under identical seed/device/batch/workers/options;
repeat seeds if assessing augmentation variability.

Validation:
```bash
python evaluate_pseudo_labels.py --help
python -m unittest discover -s tests -p test_offline_pseudo_labels.py -v
python -m py_compile evaluate_pseudo_labels.py
```
Unit tests cover checkpoint/CLI settings, weight choice, actual KNN competition
gates, disabled rescue, insufficient neighbors, predicted-class precision,
empty selections and CSV output. Full checkpoint/data inference requires your
PyTorch/torchvision FER environment and was not run in the authoring environment.
