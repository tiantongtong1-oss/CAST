# Initial-model top-k diagnosis

Base v27. No training code changes. Run diagnose_initial_model.py with a source
checkpoint and FER root; split is hard-coded to validation (val/validation).
Model eval + no_grad, pretrained=False, full strict model-state load. No optimizer,
EMA, prototypes, training loop, target train or target test data are used.
Transform matches deterministic validation: Resize256, CenterCrop224, ImageNet
normalization. DataLoader labels are ignored during inference; all predictions
are saved first, then dataset labels are joined for metrics only. Paths may
contain class folders but paths/labels are never model inputs. Do not tune on test.

```bash
mkdir -p test_logs
set -o pipefail
python -u diagnose_initial_model.py \
  --checkpoint ./models/rafdb_fer/mobilenet_v2_rafdb_fer_source_uniform_control_v27_source_best.pth \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --device cuda --batch_size 128 --workers 0 \
  --output_dir ./diagnostics/uniform_initial_v28 \
  2>&1 | tee test_logs/uniform_initial_v28.log
```

Output directory must be new/empty. Files: raw_probabilities.csv (seven class
probabilities, IDs and paths, no true labels), per_image_diagnostics.csv (joined
labels, top-3, Fear rank and top1-Fear gap), summary.json (per-class top-1/2/3
coverage and candidate precision, Fear/Sad/Neutral conditionals and gap quantiles),
confusion.csv (rows true, columns predicted). Undefined ratios use null.
Candidate precision for Fear top-2 means true Fear among ALL images with Fear in
top-2; it is not the accuracy of a corrected classifier. Top-k ranks break ties
by class index deterministically. Probability gaps describe ambiguity, not
calibrated confidence or proof of correctness.

Compare deterministic accuracy to uniform source validation 51.07%, not v26
teacher_raw single random weak-view accuracy. Backbone must match checkpoint.
No need to train first or download ImageNet weights. Checkpoint is trusted local
PyTorch serialization. Three numerical tests and CLI help/syntax checks pass.
PyTorch/data unavailable here, so full inference is not executed.
