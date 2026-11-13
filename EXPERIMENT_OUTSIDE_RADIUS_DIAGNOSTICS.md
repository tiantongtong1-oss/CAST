# Read-only outside-radius diagnostics

Base: experiment/class-medoids-shared-radius at fe51e4efd860a9358033396de77db8ba547e1675.

No selection/loss/prototype changes. Disabled by default. After radius warmup,
write all agreement-but-low-confidence rows; distinguish inside/outside/invalid.
Only outside valid rows receive the extra source/target KNN query. Its results
never enter training. No new random transforms or forwards are introduced.

Add to your normal command:

    --outside_diag_dir ./diagnostics/run_001

The directory must not exist. Optional `--outside_diag_labels` exports existing
target TRAIN labels solely for offline correctness analysis; labels never enter
selection, losses or memory updates. Do not use test labels to tune rules.
`view_js` measures two-view disagreement, not a correctness guarantee. Distance/R
uses a numerical floor; invalid centers/features are explicitly marked invalid.
Inside rows' KNN fields are not evaluated in this diagnostic pass.

Summarize one completed epoch:

    python summarize_outside_diagnostics.py diagnostics/run_001/epoch_030.csv

Reports count and optional correctness by predicted class and true class, including
outside_knn_pass. Counts are per epoch, not deduplicated across epochs. This first
version does not implement temporal memory or change training. Extra KNN queries
add runtime cost. Check CSV after one epoch before a full run.

## Keep an existing training job running

Do not switch branches or replace files in its working directory. In the existing
repository run only fetch + a separate worktree:

    git fetch origin
    git worktree add ../CAST-outside-diagnostics origin/experiment/outside-radius-diagnostics
    cd ../CAST-outside-diagnostics

Use a spare GPU or wait for the current job to finish. Use an existing completed
source-pretraining checkpoint with --pre_epochs 0, or rerun your source stage.
The training checkpoint loader is NOT a full target-stage resume facility: passing
a target checkpoint restarts scheduling/teacher/target state rather than resuming.
Worktree-relative models/logs are separate; provide your existing data/checkpoint
paths explicitly. Do not run concurrent jobs on the same GPU unless resources allow.

The supplied runtime log contains --outside_radius_cls_weight, absent from this
GitHub base. Local uncommitted changes are not captured here; preserve them and
compare before reproducing that run. This branch does not add that option.

Validation: Python syntax and static integration checks; CSV summary fixture.
PyTorch/GPU unavailable in authoring environment; tensor/training execution pending.
