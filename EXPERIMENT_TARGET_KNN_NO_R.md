# Target-only kNN, no radius ablation

Base: experiment/global-minmean-dynamic-lambda at 29eacad7ef20f6d71b30b597c2952a5ca6725437.

Admission: original/weak teacher predictions agree AND both confidences exceed the shared smoothed threshold AND target-neighbor strict majority supports the original prediction. No radius gate or medoid computation is performed. --rescue_warmup_epochs is retained only for CLI compatibility and has no effect.

Each epoch's existing threshold scan stores deterministic original-view raw features, original-view teacher argmax predictions, and dataset sample IDs for ALL valid target TRAIN images. No confidence/agreement prefilter is applied to memory; this isolates neighbor-domain replacement rather than adding a memory-quality filter. Target ground-truth labels are discarded. Validation/test images never enter the memory. Memory is frozen during the epoch and rebuilt next epoch.

Neighbor distance remains raw Euclidean, matching the source-neighbor baseline. k=11, strict majority, minimum three available neighbors; self is masked by stable sample ID before top-k. With fewer than 11 neighbors, vote over the available neighbors; fewer than 3 means reject. Duplicate memory IDs raise an error. Training/memory path ordering must match exactly.

Source labels remain used for source supervised learning and class prototypes, not target kNN. Source prototypes refresh each epoch without constructing a source medoid or radius. Threshold lambda/EMA, original/weak/strong views, classification inputs, optimizer and prototype loss are unchanged. Model suffix: _globaltau_targetknn_nor_ to avoid overwriting baseline models.

Validation: syntax and static integration checks run locally. Tensor tests cover ID self exclusion, ties, insufficient neighbors, confidence/agreement gates, invalid rows and duplicate IDs. Tensor tests require PyTorch; GPU training/data-dependent validation is not available in the editing environment.
