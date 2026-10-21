# Source resolution experiment

Based on v19 and diagnostic helpers, not the weighted v20 branch.
--source_downsample_prob defaults to 0; --source_downsample_size defaults to 48.
The proposed experiment uses probability .5. These are experimental settings, not tuned optima.
For each source training image, probabilistically downsample to short side 48, then restore
original dimensions with PIL bilinear interpolation before the original weak augmentation.
Restoring dimensions preserves crop geometry while removing detail. No grayscale or class weights.
Applies to source training during pretraining AND target adaptation; source reference/test and
all target transforms remain unchanged. Probability 0 consumes no random draw.
The inherited source-loop target-counter crash is fixed. Each evaluation logs Macro-F1,
Fear Recall, top-3 and last-rank count. Best checkpoint selection remains validation accuracy.

First train with --pre_epochs 30 --epochs 0 and no checkpoint. For a matched control, run the
same command with probability 0 and another run_name. Historical source checkpoint is context,
not a fully matched pretraining control. Then inspect the saved source model using diagnose_fear.py.
Only proceed to fixed .92 adaptation if the validation starting point is promising.
Use a separate process with --pre_epochs 0 to retain the previous fresh target optimizer setup.

Validation: four augmentation/wiring tests pass; Python syntax check passes. Real Torch/CUDA
and RAF/FER training unavailable here. No accuracy improvement is claimed.
