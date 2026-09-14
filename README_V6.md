# CAST v6 patch

A branch-ready refactor for the requested architecture:

- shared ResNet backbone;
- dual-view EMA teacher pseudo-label generation;
- DDRL class-conditional MK-MMD;
- CCDR class-volume + classifier + CATM modulation;
- module-wise console logs and JSONL metrics;
- collapse guards and a data audit script.

Run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_fer_v6.sh
```

Audit data first:

```bash
PYTHONPATH=. python tools/audit_cast_data.py \
  --source-root /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target-root /workspace/ttt/code/data/fer2013 \
  --fer-folder-order cast \
  --hash-duplicates
```

Tests:

```bash
PYTHONPATH=. pytest -q tests/test_v6_math.py
```

Important: this patch was built against the public CAST module/checkpoint naming convention. If your private/local v5 changed checkpoint keys or dataset APIs, merge those differences before running.
