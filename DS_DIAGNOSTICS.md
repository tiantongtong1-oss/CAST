# Offline source-only DS diagnostics

This adds diagnose_ds.py and a NumPy diagnostic helper without changing train.py or its acceptance rule. It builds a fresh source reference bank and class quantile radii from one checkpoint, recomputes target class thresholds using one seeded weak-view pass, and diagnoses two weak views of a fixed target subset. There is no optimizer, backward pass or training loop.

Use the same environment and dataset paths as training. No pretrained backbone download is needed. --weights auto chooses ema_teacher when saved, otherwise model. --weights model explicitly reproduces which state the training initialization command loads. Missing requested keys fail rather than silently falling back.

```bash
python -u diagnose_ds.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --weights model --backbone mobilenet_v2 \
  --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 --ds_strength_cap 0.95 \
  --source_radius_quantile 0.90 \
  --max_samples 500 --batch_size 64 --workers 0 --seed 1314 \
  --output_dir ./ds_debug/source_best_k11
```

The source_best checkpoint diagnoses initialization, not the final trained EMA Teacher or the origin of historical confidence changes. For a trained target checkpoint, replace --checkpoint and use --weights auto (or ema_teacher explicitly). Use distinct empty output directories; existing nonempty output is not overwritten.

## Output

- queries.csv: target ID/path, pseudo-label, dual confidences, adaptive threshold, agreement, high-confidence flag, region membership, rescue_candidate flag, source audit result, raw fusion support/unknown/conflict/margin for k=1,3,5,11 and the configured k, support fraction, distance-weighted support, strength-cap fraction, nearest same/other-class distance and their difference.
- neighbors.csv: per query nearest source IDs/paths, labels, distance, quality, evidence strength, capped flag and agreement with the candidate label. Up to max(11, ds_k) neighbors are exported, within ds_max_distance; ranking is shared across the k sweep. Metrics for the active gate use ds_k only.
- summary.json: checkpoint, selected weight state, checkpoint epoch, all options, selected target IDs, thresholds, reference counts and candidate pass/saturation statistics.

500 is the target sample count BEFORE candidate filtering. All sampled targets receive diagnostic output; filter queries.csv to rescue_candidate=True to study samples that would actually reach DS. The counterfactual accepted_if_rescue_enabled flag assumes warmup completed and rescue enabled. This is not a reconstruction of historical acceptance.

The k sweep shows raw normalized fusion masses even if neighbor count/conflict makes that k unavailable; inspect available_kN too. In particular k=1 cannot pass a minimum-neighbor requirement of 3. Nearest same/other distances search the entire valid source bank, not just the local neighborhood or the 11 selected references. Missing classes produce blank distances. Target ground-truth is ignored and is not exported or used in selection.

Compare raw DS support against weighted_support. Near-one DS with nontrivial unknown at small k points to accumulation. weighted_support=1 indicates local label unanimity, even without DS. A small/negative distance_gap can reveal a nearby competing class outside the top-k set. None of these scores establishes pseudo-label correctness without separate evaluation.

Repeatability requires identical checkpoint, library/device settings, batch size, workers and seed. Fresh weak augmentations/recomputed statistics can differ from a past training epoch. The source bank and threshold pass still process full datasets, so this is lighter than training but not instantaneous.

## Validation

```bash
python -m unittest discover -s tests -p test_ds_diagnostics.py -v
python -m py_compile diagnose_ds.py ds_diagnostics.py
python diagnose_ds.py --help
```

NumPy tests cover accumulation, conflict, empty neighborhoods, full-bank nearest competing-class search, strength values, CLI defaults and absence of optimizer/backward calls. Full checkpoint/data inference was not run in the authoring environment.


## v15 probability consistency diagnostics

queries.csv additionally contains prob1_class_0 through prob1_class_6 and prob2_class_0 through prob2_class_6, in CAST order (Surprise, Fear, Disgust, Happy, Sad, Angry, Neutral). js_divergence is 0.5 KL(p1||m)+0.5 KL(p2||m), m=(p1+p2)/2, using natural logarithms. It lies in [0, ln(2)] and is NOT square-root JS distance. Smaller values indicate more similar distributions, not necessarily correct predictions. confidence_difference is abs(max(p1)-max(p2)).

No JS threshold is applied to training or diagnostic acceptance. This is observational only. Keep checkpoint/weights, seed, sample count, batch size and workers identical to the preceding run. Use a new --output_dir such as ./ds_debug/target_best_k11_5000_js. Rerunning inference is necessary because previous CSVs did not contain full probability vectors. No retraining is needed.

Compare JS distributions on accepted rescue candidates first. Label-based correctness evaluation must remain offline and separate from selection/training. This branch derives from source-only v13, not target-anchor v14; train.py remains the source-only DS version.
