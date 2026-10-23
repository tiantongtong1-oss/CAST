# Concentrated Fear visual review

Run review_fear_cases.py with existing default-RGB deterministic FER validation
probability output (raw_probabilities.csv from initial-model script OR
raw_predictions.csv from feature-KNN script). All seven p_Class columns required.
No training loop, optimizer, pseudo-label writes, model/teacher/prototype updates.

Saved probabilities select up to 20 true validation Fear images incorrectly
predicted Sad/Neutral with max probability>=.8, descending confidence; up to 20
correct Fear examples descending Fear probability; 20 RAF training Fear examples
evenly spaced through existing shuffled source list. No low-confidence backfill.
These are selected examples, not unbiased dataset statistics. True validation
labels used only to choose examples for visual analysis. Never consume test data.

Full FER path/ID alignment checked. Only selected FER images are re-forwarded;
probabilities must match supplied checkpoint/default transform within tolerance,
otherwise abort to prevent stale/mismatched results. A single forward pass of RAF
train extracts reference features. Model.feature normalized cosine retrieves top3
SOURCE neighbors for chosen FER examples, no label/candidate/distance filtering.
Reference/query paths must not overlap; RAF shuffled labels verified against official
list. Actual model crop tensor is unnormalized for display (same Resize256 /
CenterCrop224 / RGB / ImageNet normalization). Original raw thumbnails shown
alongside exact model crops; resizing for display does not change inference.
Reference neighbor crops also use exact preprocessing. Captions include true
source labels and cosine similarity, query predicted probability and Fear prob.

```bash
mkdir -p test_logs
set -o pipefail
python -u review_fear_cases.py \
  --checkpoint ./models/rafdb_fer/mobilenet_v2_rafdb_fer_source_uniform_control_v27_source_best.pth \
  --predictions ./diagnostics/uniform_feature_knn_v28/raw_predictions.csv \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --device cuda --batch_size 128 \
  --count 20 --min_confidence 0.8 \
  --output_dir ./diagnostics/fear_visual_review_v28 \
  2>&1 | tee test_logs/fear_visual_review_v28.log
```

Output directory must be new/empty. Existing prediction CSV is required; no need to
repeat all prior diagnostics. Alternative predictions path is
./diagnostics/uniform_initial_v28/raw_probabilities.csv. Use matching uniform model.
Outputs: failed_fear_01..04.png, correct_fear_01..04.png,
raf_fear_reference_01..04.png (up to 20 each; five rows/page), review_manifest.json
(full paths, labels, probabilities, nearest refs and similarities), features_cache.npz
(extracted source/selected-query features, source labels and selected IDs).
Inspect failure pages FIRST, then correct/source reference examples. Upload images
and manifest for interpretation; JSON alone cannot establish alignment or semantics.
No automated claim of annotation errors/causality. Source model familiarity with
RAF train means source image predictions are not independent accuracy estimates.

Three numerical/selection/render tests + syntax/help checks pass; PyTorch/data
not available here, full feature extraction and probability check unexecuted.
