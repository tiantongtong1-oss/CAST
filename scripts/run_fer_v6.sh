#!/usr/bin/env bash
set -euo pipefail

python train_v6.py \
  --source-root /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target-root /workspace/ttt/code/data/fer2013 \
  --checkpoint ./models/cast_resnet50_v4/resnet50_rafdb_fer_source_final.pth \
  --backbone resnet50 \
  --fer-folder-order kaggle \
  --epochs 30 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --workers 10 \
  --target-lr 2e-4 \
  --backbone-lr-mult 0.25 \
  --target-lambda 0.35 \
  --ema-decay 0.999 \
  --phi 1.4 \
  --w1 4.0 \
  --w2 0.03 \
  --w3 0.1 \
  --ddrl-min-class-samples 2 \
  --ddrl-min-classes 3 \
  --affinity-warmup 5 \
  --affinity-mid-epochs 5 \
  --affinity-mid-weight 0.01 \
  --bn-recalibrate-batches 64 \
  --bn-recalibrate-momentum 0.03 \
  --debug-target-labels \
  --abort-on-collapse
