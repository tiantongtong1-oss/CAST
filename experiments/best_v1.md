Experiment:
CAST MobileNetV2 + EMA Teacher + Dual View + Stable Class-Adaptive Threshold

Repository:
tiantongtong1-oss/CAST

Branch:
improve/ema-dualview-cast

Commit:
4d6bc7a9a7c897b68aa1840894007faeda2d4bfe

Commit message:
Stabilize EMA teacher and confidence-aware thresholds

Seed:
1314

Source Dataset:
RAF-DB

Target Dataset:
FER2013

Backbone:
MobileNetV2

Command:
python train.py \
  --data1 rafdb \
  --data2 fer \
  --backbone mobilenet_v2 \
  --lr 0.001 \
  --workers 10 \
  --pre_epochs 30 \
  --epochs 30 \
  --w1 4.0 \
  --w2 0.3 \
  --w3 0.1 \
  --ema_decay 0.999 \
  --threshold_base 0.85 \
  --threshold_beta 0.5 \
  --threshold_margin 0.02 \
  --threshold_min 0.80 \
  --threshold_max 0.95

Source path:
/workspace/ttt/code/test-upload-clean/datesets/raf-basic

Target path:
/workspace/ttt/code/data/fer2013

Batch size:
128

Source pre-training epochs:
30

Target adaptation epochs:
30

Optimizer:
Adam

Learning rate:
0.001

Weight decay:
1e-4

LR Scheduler:
ExponentialLR

LR Gamma:
0.95

Classification loss weight (w1):
4.0

Feature / Affinity loss weight (w2):
0.3

Classifier modulation loss weight (w3):
0.1

EMA decay:
0.999

Threshold base:
0.85

Threshold beta:
0.5

Threshold margin:
0.02

Threshold minimum:
0.80

Threshold maximum:
0.95

Pseudo-label strategy:
EMA Teacher + Strict Dual-View Agreement

Pseudo-label acceptance condition:
1. weak view 1 and weak view 2 predict the same class
2. confidence(view1) >= class-adaptive threshold
3. confidence(view2) >= class-adaptive threshold

Student target view:
Strong augmentation

EMA update:
Fixed EMA update after every successful optimizer step.
Floating-point buffers including BatchNorm running statistics are also EMA updated.

Threshold formula:
mu_bar = mean(valid class mean confidence)

center = min(
    max(threshold_base, mu_bar + threshold_margin),
    threshold_max
)

tau_c = clip(
    center + threshold_beta * (mu_c - mu_bar),
    threshold_min,
    threshold_max
)

Source best checkpoint:
models/rafdb_fer/mobilenet_v2_rafdb_fer_ema_dualview_stable_source_best.pth

Target best checkpoint:
models/rafdb_fer/mobilenet_v2_rafdb_fer_ema_dualview_stable_target_best.pth

Best validation:
[需要从你这次最佳运行日志或 checkpoint 中读取]

Test:
[需要从你这次最佳运行日志中读取]

Checkpoint selected for final evaluation:
models/rafdb_fer/mobilenet_v2_rafdb_fer_ema_dualview_stable_target_best.pth

Checkpoint SHA256:
[需要在运行机器上计算]

Python:
3.8

PyTorch:
1.8.1

CUDA:
[GitHub 未记录实际运行 CUDA 版本]

GPU:
[GitHub 未记录实际运行 GPU 型号]

OS:
Ubuntu 16.04 LTS