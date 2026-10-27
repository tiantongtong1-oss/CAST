# CAST-equivalent classification loss (v27)
Based on experiment/global-mean-std-ema-v26 at 34df021bf690228a8a91a700b704edf08cffba2d.

Classification only is equivalent to smwanghhh/CAST train.py line 257:
mean(per_example_cross_entropy * source_and_target_mask).
This does not claim the whole training pipeline matches original CAST.

Implementation computes domains separately:
Ls = source CE mean
Lt_all = mean(target CE * learn_mask) over ALL target samples
Lcls = Bs/(Bs+Bt)*Ls + Bt/(Bs+Bt)*Lt_all

Rejected target samples contribute zero but remain in the denominator.
No target weight ramp or lambda_target is applied.
All-target-rejected batches yield source CE sum / (Bs+Bt), matching original CAST.
The existing w1 still scales Lcls in the total loss.

Mean/std/EMA threshold settings remain:
candidate = clip(mean(min_confidence) + 0.5*population_std, 0.85, 0.99)
tau = 0.8*previous_tau + 0.2*candidate (first scan uses candidate).
Agreement, rescue, stable promotion, prototype and other loss settings retain base behavior.
The earlier v26 document describes the historical separate-mean/ramp loss;
this v27 document supersedes that classification description.
Default LR 0.001, workers 10, run_name global_mean_std_cast_ce_v27.

Tests:
python -m unittest discover -s tests -p test_cast_classification.py -v
python -m unittest discover -s tests -p test_global_mean_std_ema.py -v
Numerical tests execute the actual CE block with a NumPy tensor shim and compare
against original CAST's masked mean across equal/unequal batch sizes and masks.
No full PyTorch training was run in the authoring environment.
