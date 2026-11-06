# Fusion with normalized source Q90 radius fallback

Admission: original/weak agreement AND (both confidences pass threshold OR normalized original query distance < normalized source medoid Q90 radius) AND fused kNN argmax matches pseudo-label AND fused support >= --knn_support_threshold (default .7).

R is rebuilt from normalized source features each epoch; strict d<R, no .7 radius multiplier. Radius fallback starts at epoch index --rescue_warmup_epochs (default 3). Set it to total epochs to disable fallback. Both paths must pass the unchanged 0.5 source/0.5 target soft fusion. Radius_Rescue_Num counts only radius candidates that pass fusion. Checkpoint suffix q90r avoids overwriting no-R models. Parent no-R documentation describes historical behavior.

Syntax/static checks performed; PyTorch tensor and GPU training unavailable in editing environment.
