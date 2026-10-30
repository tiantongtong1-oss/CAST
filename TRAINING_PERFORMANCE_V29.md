# v29 training progress and transfer optimization

CSV diagnostic columns and acceptance rules are unchanged. Query labels are
copied once and reused by the NumPy audits; accepted/radius/vote masks are copied
as a batch before constructing rows. CSV row iteration never indexes GPU tensors
or calls item(). Source prototypes used by CPU competition are cached per epoch.

Source images/labels are moved once to the training device. Images, pseudo labels,
and source/target anchor masks are concatenated on that device, with no strong.cpu(),
pseudo_targets.cpu(), or anchor_idx.cpu() round trip for the training input.
CPU copies required by NumPy KNN and diagnostic statistics remain.

Source pretraining and target training print progress every20 batches and at the
last batch, including elapsed time, window time, seconds/batch and remaining time
for the CURRENT epoch's batch loop (not validation or future epochs).
PhaseWall fields report accumulated host wall time during that window: data,
transfer, teacher_gate, knn_competition, rescue_csv, roles_stats, train_update.
CUDA is asynchronous: these are coarse host phase timings including any waiting
at existing synchronization points, not precise CUDA-kernel profiling. No new
per-stage CUDA barriers are inserted. Existing scalar loss reads complete GPU
training work before batch reports. Source training uses data/transfer/train.
Epoch setup additionally announces threshold scan, KNN refresh, radius refresh
and total setup completion, so setup is distinguishable from batch processing.

Run commands and v29 class-balanced quantile radius are unchanged. Tests cover
actual CSV row assembly, actual tensor-concatenation expressions, deterministic
progress frequency/ETA, and existing training regressions. PyTorch/CUDA is absent
in the development environment; real GPU speed and memory remain unmeasured.
