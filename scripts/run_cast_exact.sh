#!/usr/bin/env bash
set -euo pipefail

python train_cast_exact.py \
  --source-root /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target-root /workspace/ttt/code/data/fer2013 \
  --backbone resnet50 \
  --fer-folder-order kaggle \
  --source-epochs 30 \
  --epochs 30 \
  --lr 0.001 \
  --lr-gamma 0.95 \
  --weight-decay 0.0001 \
  --w1 4.0 \
  --w2 0.3 \
  --w3 0.1 \
  --phi 1.4 \
  --workers 10 \
  --selection-split test
