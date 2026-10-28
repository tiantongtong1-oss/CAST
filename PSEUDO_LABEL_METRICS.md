# Target-training pseudo-label quality diagnostics

Based on experiment/global-mean-std-cast-ce-v27 at
37f0d8e55f56ce41de68f0884fa0d04ee4db4996. No threshold or loss changes.

Every target epoch now reports accepted count, correct count and pseudo-label
accuracy for each predicted class and ALL, for five stages:

| stage | actual selection |
| --- | --- |
| confidence | direct dual-view confidence acceptance |
| rescued | rescue acceptance, including pending promotion |
| final | confidence OR rescued; not necessarily all supervised |
| supervised | actual learn_mask used by target CE and prototype consistency |
| anchors | stable anchors used by MMD and target prototype updates |

Read `supervised` first for the training-label quality. Stage sets overlap;
do not add counts across all stages. Accuracy is correct/accepted, grouped
by PREDICTED class, not recall. No accepted samples means N/A in the console
and an empty CSV field, not zero accuracy. ALL is computed from total correct
and accepted counts, not an average of class or batch accuracies.

The CSV is `./models/rafdb_fer/<run_name>_pseudo_label_metrics.csv` and has
epoch, stage, class_id, class_name, accepted_count, correct_count,
pseudo_accuracy (fraction 0..1), scanned_count. There are 40 rows per epoch:
five stages times seven classes plus ALL (class_id=-1). Epochs start at zero.
The CSV is initialized afresh on each invocation, so use a distinct run_name.

Ground truth is read only by the diagnostic accumulator through sample IDs
from the exact target_train dataset. FER.label already converts native FER IDs
to CAST order: Surprise, Fear, Disgust, Happy, Sad, Angry, Neutral. Keep your
existing FER directory convention (native numeric FER IDs). No Dataset format
change or extra inference pass is needed. The helper checks ID uniqueness and
complete per-epoch coverage; the existing shuffled target loader visits every
image once, including the final partial batch. It does not consume random
numbers or modify tensors, masks, thresholds, weights, losses or memories.

These are online, actual-training-pass statistics: Teacher parameters and
stable roles evolve during an epoch. They are not a frozen-checkpoint scan or
a reconstruction of previous runs. Existing checkpoints do not contain these
historical metrics; rerun training to collect them. Do not use these diagnostic
target-training labels to drive selection, gradients or checkpoint selection.

## Run

Keep all your previous experiment options and change only run_name, e.g.:

```bash
python -u train.py \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 30 --epochs 30 \
  --workers 10 --lr 0.001 \
  --run_name global_mean_std_cast_ce_v27_pseudo_metrics
```

If reusing a source checkpoint, supply your existing --checkpoint path and
--pre_epochs 0. The current training entry point loads model weights, not a
full target-stage optimizer/role/threshold resume; this starts a new target run.

```bash
python -m unittest discover -s tests -p 'test_pseudo_label_metrics.py' -v
python -m unittest discover -s tests -p 'test_cast_classification.py' -v
python -m unittest discover -s tests -p 'test_global_mean_std_ema.py' -v
```

Full data/GPU training must be run in your FER environment.
