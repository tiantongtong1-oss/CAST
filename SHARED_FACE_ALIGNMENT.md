# Shared RAF-DB / JAFFE alignment

prepare_rafdb.py imports choose_face, align_face and TEMPLATE directly from
prepare_jaffe.py: no duplicate template. Both use the same RetinaFace dependency,
threshold .9 and 256x256 five-point similarity alignment. This unifies our pipeline,
not a claim to reproduce CAST's unpublished alignment template.

Keep the exact existing JAFFE processed pool (if created with the default script).
RAF auto prefers Image/original; if that directory is absent, uses Image/aligned
and records input_kind=aligned (secondary alignment). Explicit --input_kind original
or aligned overrides detection. Missing/ambiguous inputs fail before inference.
Largest detected face is selected; inspect multi-face records in alignment_report.csv,
especially if original RAF scenes contain multiple people. Largest does not guarantee
the annotated face. If original scenes mismatch, consider the labeled aligned inputs
as a separate declared secondary-alignment experiment; never silently switch by image.

New output root has Image/aligned/*_aligned.png and an unchanged copy of
EmoLabel/list_patition_label.txt. All original split records are processed (normally
12271 train + 3068 test). No failed detection is dropped or replaced by raw input.
On interruption/failure rerun same command with --resume; completed matching-job outputs
are reused, failed/missing files retried. Do not modify input images during a job.
Incomplete metadata prevents training; inspect report if any failure remains. A global
threshold change requires a new output root and matching JAFFE processing for strict
metadata equality. Do not lower thresholds blindly for mislabeled detections.

## Update and prepare source

```bash
cd /workspace/ttt/code/CAST
git fetch origin
git switch experiment/source-ds-js-jaffe-v19
git pull --ff-only origin experiment/source-ds-js-jaffe-v19
CUDA_VISIBLE_DEVICES='' .venv_jaffe_align/bin/python prepare_rafdb.py \
  --input /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --output /workspace/ttt/code/data/rafdb_retinaface_256 \
  --input_kind auto --resume
```

Use the already installed preprocessing venv and cached RetinaFace weights.
Processing all RAF images is much slower than 213 JAFFE images; resume is supported.
Output is local only; no images or labels are uploaded by the script.

## Retrain source, then adapt target (no old CAST checkpoint)

```bash
conda activate cast
cd /workspace/ttt/code/CAST
python -u train.py \
  --data1 rafdb --data2 jaffe --backbone mobilenet_v2 \
  --source_path /workspace/ttt/code/data/rafdb_retinaface_256 \
  --target_path /workspace/ttt/code/data/jaffe_retinaface_256 \
  --require_shared_alignment \
  --pre_epochs 30 --epochs 30 --target_steps_per_round 200 --workers 0 \
  --lr 0.001 \
  --model_dir /workspace/ttt/code/CAST/jaffe_models \
  --log_dir /workspace/ttt/code/CAST/jaffe_logs \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --immediate_prototype_update \
  --ds_neighbor --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 --ds_strength_cap 0.95 \
  --ds_support_threshold 0.6 --ds_margin_threshold 0.1 \
  --ds_uncertainty_threshold 0.3 --ds_conflict_threshold 0.5 \
  --rescue_js_threshold 0.01 --rescue_js_start_epoch 20 \
  --run_name jaffe_sharedalign_source30_fixed6000_js4000
```

No --checkpoint: retrain source, keep weights in memory and reset target optimizer.
Default ImageNet backbone initialization remains unchanged. Target budget stays 6000
updates with JS after 4000; all algorithm settings and final student evaluation remain.
Record source train and test domain preprocessing changes when comparing with old runs;
different source training inputs necessarily produce a different source checkpoint.

Loader remains compatible with legacy RAF aligned JPG roots. New PNG naming is enabled
only by generated alignment_metadata.json. --require_shared_alignment checks complete
metadata, method/template/size/threshold equality, not detector package version or actual
image quality. Use the same venv and detector weights for both datasets.

Tests cover input preference, preserved labels/splits/PNG paths, and metadata rejection.
Existing geometry, JAFFE and scheduling tests retained. Detector inference and full GPU
training were not executed in the authoring environment.
