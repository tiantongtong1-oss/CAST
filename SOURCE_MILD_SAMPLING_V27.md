# Source-pretraining-only mild sampling (v27)

Base: v26 92b1dcffc7a8ea890731ea9f7798cec2372e5039 (v25 filtering retained).
Branch: experiment/source-mild-sampling-v27.

Opt-in --source_pretrain_sampling mild --source_sampling_power 0.5.
Per-image weight = source class count ** (-power). WeightedRandomSampler uses
replacement=True, num_samples=len(source_train), private deterministic seed 1341.
Expected class mass is n_c**(1-power), not equal classes at power=.5. RAF counts
[1290,281,717,4772,1982,705,2524] yield expected draws
[1633.3,762.3,1217.6,3141.3,2024.5,1207.4,2284.6]. Counts fluctuate per epoch;
Actual_Source_Draws reports actual exposures. Repeats do not add unique images.
Uniform default reuses original shuffled loader. No loss class weighting.
Sampler affects source pretraining ONLY. Target-stage source batches retain
original shuffle and full source prototype/KNN reference loaders are unchanged.
Source affinity sees a different batch distribution naturally due to resampling;
its formula and coefficient remain unchanged. No augmentation changes.

## 1. Train a new source checkpoint, no old checkpoint initialization

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --backbone mobilenet_v2 \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --pre_epochs 30 --epochs 0 --source_only --lr 0.001 \
  --source_pretrain_sampling mild --source_sampling_power 0.5 \
  --run_name source_mild_sampling_v27 \
  2>&1 | tee test_logs/source_mild_sampling_v27.log
```

Without --checkpoint, Networks uses its normal ImageNet pretrained backbone (weights
must be cached or downloadable). Does not start from a random backbone. Source-only
returns after source-best loading and validation; no target optimization, Teacher,
prototype initialization, stage diagnosis or test evaluation occurs. Existing
pipeline still constructs target datasets/loaders; labels do not enter training.
Source-best selection remains original validation ACCURACY, unchanged for isolation.
Compare Fear precision/recall, UAR and macro-F1 as well as accuracy. For a fair source
control, repeat same command with --source_pretrain_sampling uniform and run_name
source_uniform_control_v27. Do not compare only to an old checkpoint of unknown
training provenance. Test labels must not guide sampler parameters or selection.

Output: models/rafdb_fer/mobilenet_v2_rafdb_fer_source_mild_sampling_v27_source_best.pth

## 2. Optional separate target adaptation from the NEW source checkpoint

Use previous v26 target command, but --checkpoint points to the new source file,
--pre_epochs 0 --epochs 30, omit --source_only and use --source_pretrain_sampling
uniform (default). Keep --run_name target_from_mild_source_v27. Sampling mild with
pre_epochs=0 is rejected to prevent accidentally doing an ineffective experiment.
Use same target configuration for uniform and mild source models. Target epoch-0
teacher_raw is the direct comparison of new initial pseudo-label quality.

Alternatively one invocation with pre_epochs=30 epochs=30 and mild sampling, no
source_only, performs both stages; target-stage loader is still unweighted.
Do not pass v23 source_class_weight flags or immediate_prototype_update.

Validation: three NumPy sampling tests pass; Torch sampler test skipped because
PyTorch unavailable. Syntax/diff/CLI checks pass. Full source training unrun;
no performance improvement claimed.
