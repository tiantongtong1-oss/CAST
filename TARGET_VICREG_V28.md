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

## Stabilization options

--freeze_backbone_bn: after model.train each epoch, call eval on BN modules under
model.feature only. Running mean/variance remain fixed, affine parameters still
train. Source classifier BN and projector BN remain train-mode. Default disabled
retains prior behavior. This is stats freezing, not freezing the whole backbone.

--grad_log_interval 100: sample at global steps 0,100,..., log gradient L2 norms
of weighted source CE and weighted SSL on SHARED model.feature parameters, ratio,
and cosine to gradient_diagnostics.csv. autograd.grad retains graph, does not
populate/modify parameter.grad; normal total.backward follows. Includes both loss
weights; excludes projector/classifier-only parameters for comparable gradients.
Zero norms yield undefined cosine/ratio as blanks. Default 0 disables diagnostics.
Sampled extra backwards add overhead, not every step. Negative cosine signals
conflicting gradients on that sampled batch, not proof of whole-run conflict.

Suggested short stability run FROM ORIGINAL uniform source model: epochs=2,
ssl_weight=.01, source_weight=1, lr=.0001, freeze_backbone_bn, grad_log_interval=100,
NEW output_dir target_vicreg_bnfreeze_w001_v28. This changes BN stats and SSL
strength together for a practical stability trial, NOT a causal isolation of the
two effects. No universally correct gradient ratio threshold; compare validation
MacroF1 and Fear precision/coverage rather than loss scale alone. Use matching
ssl_weight=0 + BN freeze as an optional baseline if attribution is later needed.

Evaluate previous best.pth separately using diagnose_initial_model.py. Best may
change if the previous job continues; consult checkpoint epoch/current log before
calling it epoch0. Preserve previous run directory. Do not initialize new trial
from degraded last.pth. Full diagnostic eval has no training/state updates.

Added three Torch tests for BN stats/affine gradients, weighted gradient ratio,
nonmutation, and opposing/zero gradients. All skipped here (no PyTorch); run on
training host. Syntax/help and two unlabeled-data tests pass. Full stability trial
not executed; no guarantee of performance recovery.
