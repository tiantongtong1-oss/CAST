# Equal-update target adaptation

Optional `--target_steps_per_round 225 --epochs 30` gives exactly 6750 target
optimizer updates for either FER or JAFFE. Default 0 preserves native-epoch behavior.
The existing epoch flags and logs now denote **scheduling rounds** in fixed mode;
each round spans exactly 225 updates, independent of the target dataset size.

Default event mapping (number of completed target updates):

| Event | Completed updates before activation/change |
| --- | ---: |
| Source DS refresh, target threshold refresh | 0, 225, 450, ... |
| Source radius refresh (interval 1) | 0, 225, 450, ... |
| Radius rescue starts | 675 |
| Prototype gradient loss starts (weight .02 for default max .1) | 675 |
| Prototype weight becomes .1 | 1575 |
| JS starts at round 20 | 4500 |
| LR first decay | 225 |
| End | 6750 |

Thus JS applies to updates 4501-6750 (human one-based counts), not after only
40 JAFFE updates. Prototype loss retains the existing five-window staircase ramp
(weights .02,.04,.06,.08,.1), not a new continuous linear-per-step rule.
Radius warmup, prototype warmup, JS threshold, losses, EMA update and source DS
remain unchanged; only the target time unit becomes a fixed optimizer-update window.

Target DataLoader exhaustions recreate the iterator, reshuffle and resample augmentations.
No batch cache/itertools.cycle is used. Iterators carry across scheduling windows,
so a window boundary does not discard unconsumed target batches. Source batches
are sampled as in the baseline (source iterator starts each round).

Anchors retain the baseline semantics of consecutive scheduling rounds: repeated IDs
inside a round cannot increment the promotion streak multiple times. This is not a
claim of equal numbers of independent observations or equally reliable anchors.
Prototype updates still happen on each accepted batch, including repeat exposures.
Statistics in fixed-round logs/DS CSV are observation records, not unique target counts.

Equal updates do not mean equal unique images, examples or compute: JAFFE repeatedly
exposes 213 images with fresh augmentation, and its remainder batch is 85 images;
FER has far more distinct images. No speed/accuracy gain is guaranteed.

Checkpoint args and target_update_schedule record the budget and event locations;
Completed_Updates is logged each round. These checkpoints still are initialization
artifacts, not exact resume support.

## Matched initialization

Use the same fixed RAF source model for both datasets, e.g. the already generated:

`/workspace/ttt/code/CAST/jaffe_models/mobilenet_v2_rafdb_jaffe_jaffe_retinaface_source30_target30_js20_source_last.pth`

Both commands use --pre_epochs 0 and this checkpoint. Do not initialize FER using a
different FER-selected source checkpoint when claiming matched initialization.
If avoiding checkpoint reads is required, retrain source for each run, but that is
not guaranteed to produce identical initialization because dataset construction can
affect random-number states. Shared source weights are the recommended controlled setup.

JAFFE retains last-round student evaluation, with no labeled validation selection.
FER retains validation-best student selection then held-out test. Report this protocol
difference; equal schedules do not turn JAFFE transductive evaluation into held-out testing.
Existing old FER scores are not a substitute for rerunning with the declared schedule.

## Commands

Common flags: --pre_epochs 0 --epochs 30 --target_steps_per_round 225
--immediate_prototype_update --ds_neighbor --ds_k 11
--rescue_js_threshold .01 --rescue_js_start_epoch 20
Other defaults retain the current v19 settings. Use a fresh run_name.

Checks: python -m unittest discover -s tests -p 'test_update_schedule.py' -v
and the existing test_jaffe tests plus py_compile. Tests cover repeated dataset passes,
fresh reads, cross-window continuation, empty loaders and dataset-independent timing.
The authoring environment has no torch/CUDA or datasets; full training remains untested.
