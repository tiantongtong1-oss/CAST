# Target feature adaptation: source CE + target VICReg-style objective

New standalone entry adapt_target_vicreg.py, existing train.py unchanged.
Reference: Bardes et al., VICReg, ICLR 2022, https://arxiv.org/abs/2105.04906
and https://github.com/facebookresearch/vicreg . This is a compact single-device
transfer adaptation experiment, NOT a reproduction of the ImageNet benchmark.

Initialize uniform source checkpoint strictly. Target loader discovers ONLY
train/*/*.jpg, does not parse label folder names, never loads/returns target labels.
Each image returns two independently sampled mild weak views. Source RAF train
uses true labels, original uniform shuffle. Train and validation paths disjoint.
FER val labels used for evaluation/model selection only; no test dataset accessed.

Features = model.feature (512 for MobileNetV2), both branches receive gradients.
Compact projector Linear(512,512)-BN-ReLU-Linear(512,512). No final BN or L2 norm.
VICReg: invariance MSE; variance = mean ReLU(1-sqrt(population_var+1e-4)), averaged
across both views; covariance = sum squared offdiagonals of both sample covariance
matrices divided by D. Within-loss coefficients 25/25/1. Population std estimate is
an explicit local implementation choice. Drop-last, B>=2 prevents undefined sample
covariance. This regularizes collapse but does not guarantee useful semantics or
Fear recall; do not claim the experiment has already solved the problem.

Loss = source_weight*source_CE + ssl_weight*VICReg. Defaults 1 and .1, LR1e-4,
Adam weight decay1e-4, fixed LR, 10 epochs. Feature backbone BN sees concatenated
source + target views; final classifier BN sees SOURCE rows only. Dropout stays
active in train mode; validation encoder geometry uses eval mode to avoid mistaking
dropout variance for useful feature diversity. No EMA teacher, pseudo labels,
prototype loss, affinity/MMD, source class weighting/resampling, or label repair.
Existing uniform source classification weights remain trainable.

```bash
mkdir -p test_logs
set -o pipefail
python -m unittest discover -s tests -p test_vicreg_adaptation.py -v
python -u adapt_target_vicreg.py \
  --checkpoint ./models/rafdb_fer/mobilenet_v2_rafdb_fer_source_uniform_control_v27_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --device cuda \
  --epochs 10 --batch_size 64 --workers 4 --lr 0.0001 \
  --source_weight 1.0 --ssl_weight 0.1 --projection_dim 512 \
  --output_dir ./models/rafdb_fer/target_vicreg_v28 \
  2>&1 | tee test_logs/target_vicreg_v28.log
```

New/empty output_dir required. If OOM, lower batch_size to32; record change because
variance/covariance estimates depend on batch size. A forward processes 3*batch_size
images. Existing model checkpoint format is reused via key model (projector is
training-only); diagnostic scripts load best.pth/last.pth directly. Source-only
checkpoint loading is initialization, not exact resume. Best checkpoint includes
initial baseline epoch=-1; best selection is validation Macro-F1, not Fear-only or
accuracy. If training never beats baseline, best.pth remains initial model.
last.pth stores last adapted model. JSON per epoch includes all class top1/2/3
metrics and eval encoder std/normalized std/average pairwise cosine. Training CSV
includes three losses and projected std monitoring. No adaptation claim can be
made until validation metrics improve without unacceptable precision loss.

Evaluate the saved best.pth using diagnose_initial_model.py with a NEW output dir.
Do not immediately feed it into full hard-label self-training: first inspect initial
Fear recall/top2 coverage and macro-F1 vs uniform. New checkpoint has trained on
unlabeled FER train, so describe it as adapted, not source-only model.

Optional matched loss ablation: ssl_weight=0 and a different output_dir. This still
exposes backbone BN to target images and is NOT pure source-only fine-tuning.

Validation here: two file/label-isolation tests + three top-k numerical tests pass;
three VICReg tensor/gradient tests skipped because PyTorch unavailable. Syntax and
CLI help pass. Full training, CUDA memory requirements and gradients not executed
here. Run test_vicreg_adaptation.py on training host before training; all five tests
should pass. No target true labels used in optimization or batch sampling.
