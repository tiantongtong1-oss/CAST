# Feature-based source neighbor comparison (v28 addition)

Run diagnose_feature_knn.py on the same source uniform checkpoint. Adds standalone
scripts only; training and diagnose_initial_model.py are unchanged.

Reference: RAF training features + true source labels. Queries: FER validation
only. Forward uses labels=None, model.eval and no_grad, pretrained=False, strict
full model load. No target training/test data, optimizer, EMA or prototype updates.
All outputs are decided and raw_predictions.csv saved BEFORE evaluation joins
FER labels. Paths can contain class folders, but paths are not model inputs.

Feature: existing FeatureHook(model.feature), same as training KNN bank. L2
normalize, Euclidean distance. Deterministic Resize256/CenterCrop224/ImageNet
normalization, no random augmentation. GPU block matrix multiplication by default;
only query-chunk x source_count distances are retained. Use --device cpu if needed.

Methods:
- classifier_head: logits argmax.
- source_knn: global nearest K=11, unweighted true source label majority.
- classwise_distance: nearest K=11 within EACH source class; smallest average
  Euclidean distance wins. Every class must have >=K valid references, else error.
No candidate labels, distance cuts, confidence filtering or rejection. Prediction
may be any of seven classes. Stable source-order tie ranking; class ties use
lowest class ID. Small floating CPU/GPU differences may affect near ties.
Per-class equal K reduces count dominance but does NOT remove density/diversity
bias and is not a claimed optimal classifier.

```bash
mkdir -p test_logs
set -o pipefail
python -u diagnose_feature_knn.py \
  --checkpoint ./models/rafdb_fer/mobilenet_v2_rafdb_fer_source_uniform_control_v27_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --device cuda --batch_size 128 --workers 0 \
  --k 11 --query_chunk 64 \
  --output_dir ./diagnostics/uniform_feature_knn_v28 \
  2>&1 | tee test_logs/uniform_feature_knn_v28.log
```

Output directory must be new/empty:
- feature_summary.json: all three methods' Accuracy/UAR/Macro-F1, per-class
  precision/recall/F1, confusion, Fear recovery/loss vs head.
- raw_predictions.csv: no target true labels; IDs, image paths, head/KNN/classwise
  predictions, seven probabilities, seven votes, seven mean distances.
- per_image_evaluation.csv: true labels joined only after inference, three outputs.
- confusions.csv: method, true class, seven predicted-class counts.

new_fear_precision uses ONLY predictions newly changed to Fear vs head. Also
report lost_head_correct_fear and net_correct_fear_change. Undefined ratios=null.
Check classifier_head accuracy against deterministic uniform validation 51.07%.
Checkpoint backbone must match. No ImageNet download or retraining needed.
Seven numerical tests + syntax/help checks pass; full PyTorch/GPU/data inference
not executed in editing environment. No performance guarantee.

## RAF official test queries (addition)

```bash
python -u diagnose_feature_knn.py \
  --checkpoint ./models/rafdb_fer/mobilenet_v2_rafdb_fer_source_uniform_control_v27_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --query_dataset raf_test \
  --backbone mobilenet_v2 --device cuda --batch_size 128 --workers 0 \
  --k 11 --query_chunk 64 \
  --output_dir ./diagnostics/uniform_raf_test_v28 \
  2>&1 | tee test_logs/uniform_raf_test_v28.log
```

RAF train remains the only source reference set; RAF official test is query-only.
FER is not instantiated in raf_test mode and target_path is unnecessary. Default
fer_val keeps old behavior and requires target_path. RAF query wrapper adds IDs
without changing dataset.py or its train/test behavior. Query forward task=source
in RAF mode. All three predictions/evidence saved before query labels join metrics.

Audit verifies complete official split basename membership and shuffled path-label
alignment for RAF train and RAF test. Resolved image paths must be disjoint and
unique within each split. This is a PATH check, not a content duplicate hash audit;
it does not independently establish checkpoint training provenance. It is assumed
the supplied source-only model was trained on official RAF train, not test.
Audit records RAF 1..7 -> class names and checks expected FER mapping, verifies
both current loader implementations explicitly contain BGR->RGB conversion,
and records shared deterministic transform. It does not infer semantic annotation
quality from image content. No training code changed.
Summary includes dataset_audit and split=RAF official test only. Compare per-class
recall/F1 along with precision due to differing RAF/FER class frequencies. Repeated
use to tune models makes RAF test a development evaluation set, not untouched
final test evidence. Ten data-audit/retrieval/metric tests pass plus syntax/help
checks; no end-to-end PyTorch/data execution in editing environment.

## RGB versus grayscale input diagnostic

--color_mode rgb (default) or gray controls query preprocessing. Gray adds
Grayscale(num_output_channels=3) AFTER CenterCrop and BEFORE ToTensor/normalization.
--reference_color_mode same (default) applies same mode to RAF reference images.
Thus gray mode re-extracts references and queries; no model retraining. Classifier
head comparison is independent of reference mode. To isolate query-only changes
in retrieval, use --reference_color_mode rgb in BOTH runs. Changing references
and queries together measures combined effect, not only query color dependence.
ImageNet normalization stays identical across modes: grayscale channels are equal
before normalization, not after different channel means/stds are applied.

Run four evaluations (query datasets raf_test/fer_val x modes rgb/gray) using same
checkpoint, K, source data and deterministic geometry. Original RGB results should
match prior diagnostics. Do not infer improved train-time grayscale augmentation
from a test-time change alone. FER images may already be gray in RGB storage;
input_audit.json measures sample channel differences to check that possibility.

--audit_images 32 samples first 32 loader paths per domain (0 disables). Saves
input_audit.json with raw dimensions, stored mode, mean RGB channel spread and
fraction spread>2. input_preview.jpg shows first six reference/query images as
raw thumbnail, center crop and gray crop. Preview illustrates geometry only (PIL
resizing may differ from model transform interpolation); it is not an exact model
tensor. No labels or model decisions determine sample selection. These are small
nonrandom samples, not exhaustive face alignment/annotation checks; upscaling a
48x48 source image does not recover detail. Path disjoint audit is unchanged.
Output directory must be new/empty for each run. No model/state updates or target
train/test queries in fer_val mode. Existing RAF official test option unchanged.
Nine numerical/image/data/retrieval tests and syntax/help checks pass; full
PyTorch color-transform/inference not executed here. Inspect previews manually.

## Resolution bottleneck diagnostic

--query_lowres_size 0 (default): native query input, original results unchanged.
--query_lowres_size 48: PIL bilinear resize to 48x48 BEFORE the existing
Resize256 -> CenterCrop224 -> optional Grayscale3 -> ToTensor/normalization.
--reference_lowres_size 0 (default): keep source training reference images native.
Set reference size explicitly only for a separate matched-reference experiment.

For a controlled RAF test, run query_lowres_size 0 and 48 with query_dataset
raf_test, color_mode rgb and reference_color_mode rgb, reference_lowres_size 0.
Use separate output directories. Reference library thus stays unchanged; both
classifier and retrieval methods see only the query resolution intervention.
No source/target retraining, teacher, optimizer or prototype writes. Test labels
join after predictions as before. Lowres0 results should reproduce RAF RGB baseline.

Summary dataset_audit.resolution_modes records both sizes. 0 means native, not
zero pixels. When audit_images>0 and query lowres>0, resolution_preview.jpg shows
first six query images' original and bottleneck crops. It illustrates geometry;
preview PIL interpolation may not be pixel-identical to model's final resize.
input_audit.json dimensions describe original files, NOT transformed tensor size.
Images are not overwritten. Downsampled RAF does not replicate FER annotation,
face alignment, grayscale style or all camera artifacts; a performance change is
resolution sensitivity evidence, not proof of the whole domain-gap cause.
Four resolution tests + four retrieval tests, help/syntax checks pass. Full model
inference not executed in editing environment (PyTorch/data unavailable).
