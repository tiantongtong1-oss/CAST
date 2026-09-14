#!/usr/bin/env bash
set -euo pipefail

python train_v6.py \
  --source-root /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target-root /workspace/ttt/code/data/fer2013 \
  --checkpoint ./models/cast_resnet50_v4/resnet50_rafdb_fer_source_final.pth \
  --backbone resnet50 \
  --fer-folder-order kaggle \
  --epochs 15 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --workers 10 \
  --target-lr 2e-4 \
  --backbone-lr-mult 0.25 \
  --target-lambda 0.35 \
  --target-lambda-ramp 4 \
  --ema-decay 0.999 \
  --phi 1.4 \
  --threshold-cap 0.95 \
  --pseudo-keep-start 0.35 \
  --pseudo-keep-end 0.55 \
  --source-correction-alpha 0.5 \
  --source-correction-min 0.85 \
  --source-correction-max 1.5 \
  --pseudo-class-weight-max 1.25 \
  --class-weight-correction-gate 1.10 \
  --w1 4.0 \
  --w2 0.01 \
  --w3 0.1 \
  --ddrl-min-class-samples 2 \
  --ddrl-min-classes 3 \
  --affinity-warmup 5 \
  --affinity-mid-epochs 5 \
  --affinity-mid-weight 0.005 \
  --bn-recalibrate-batches 0 \
  --early-stop-patience 4 \
  --debug-target-labels
