# Uniform shared Q90 gate before fused kNN

Parent experiment/class-medoids-shared-radius at fe51e4efd860a9358033396de77db8ba547e1675.
Epochs below --rescue_warmup_epochs: unchanged confidence+agreement then fusion.
At/after warmup (default index 3): agreement AND normalized distance to pseudo-label class medoid < shared class-balanced Q90 R AND unchanged fusion argmax/support gate.
High-confidence candidates no longer bypass R. Low-confidence radius candidates are retained. Thus confidence no longer determines eligibility after warmup, but remains a diagnostic route classification; it is not confidence AND R.

Teacher pseudo-labels, two-view-mean threshold/EMA, k=11 per domain, equal soft fusion, source/target L2 normalization, Q90, Student strong augmentation and prototype weight CLI are unchanged. Log original confidence candidates separately from gated confidence and confidence rejected by R. Before warmup gated confidence means passthrough; R rejection is zero.
New model suffix uniformgate avoids overwriting baseline checkpoints.

Validation: production mask helper tested exhaustively over all 16 boolean cases without PyTorch, including warmup, high-confidence R rejection, low-confidence R acceptance and disagreement rejection. Syntax/static integration checks; no GPU training or tensor execution in editing environment.
