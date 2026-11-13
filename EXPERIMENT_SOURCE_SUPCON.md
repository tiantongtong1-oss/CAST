# Source-only balanced-batch supervised contrastive fine-tuning

Based on balanced-head-crt 2505960. Existing CAST training is untouched.
Independent train_source_supcon.py loads checkpoint['model'], retains its head,
fine-tunes backbone + head with CE + weight * supervised contrastive loss.
A separate 128-dimensional projection MLP is discarded for inference. This is
balanced-batch SupCon, NOT a complete reproduction of BCL/BPaCo (no class-complement
centers). Each batch samples 8 unique source rows per class where available and two
independently augmented views: 56 images, 112 views. All seven classes appear equally.
Insufficient class rows use replacement. Cropping/flipping only, no severe erasure.
Default 20 epochs, Adam lr=1e-4, contrast weight=.1, temperature=.1. BN updates normally
in this representation experiment, unlike frozen-head cRT. No affinity/CCDR/prototype
losses or target samples. Default seeded stratified 10% source head-stage holdout;
loaded backbone may already have seen it, so this is not unseen-source validation.
Best selected solely by source-val Macro-F1. Outputs source_best.pth/source_last.pth,
metrics.jsonl, source validation precision/recall/F1, split paths. Fresh output dir only.

Required matched control: identical seed/config with --contrast_weight 0 and a new
directory. Both runs use the same balanced batches, views, projection forward and
schedule; thus comparison isolates added contrastive gradients, not continued
training or sampling. Compare multiple seeds before claiming stable improvement.
Evaluate both with scan_candidates.py --weights model; target labels evaluation only.
GPU 0 must be idle. Lower --per_class to 4 for memory, but match BOTH conditions.
Do not overwrite the original source checkpoint or replace the CAST teacher yet.

Tests: syntax and sampler split logic passed locally. Added tensor finite-gradient,
permutation, no-positive and class-balance tests. No torch/CUDA in authoring environment,
so tensor tests/full GPU training not executed. Server command:
python -m unittest discover -s tests -p test_source_supcon.py -v
