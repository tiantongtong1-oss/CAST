# Global mean/std/EMA confidence threshold
Based on experiment/global-quantile-higher-v25, commit f71c0c7774b830c57f64eeca12b85fae76fa6044.

Every target epoch begins with a complete teacher scan of the target training split.
For predictions agreeing across two independent weak views, pool one score per image:
s = min(conf1, conf2). Target labels are not read.

candidate = clip(mean(s) + threshold_std_scale * population_std(s), threshold_min, threshold_max)
tau = candidate on the first valid scan; otherwise:
tau = threshold_smoothing * previous_tau + (1 - threshold_smoothing) * candidate

Defaults: std scale 0.5, smoothing 0.8, minimum 0.85, maximum 0.99.
An empty scan holds the previous tau; an initial empty scan uses threshold_base=0.85.
All seven classes share tau, using the existing repeated-vector selector interface.
Agreement and min confidence >= tau remain required for direct acceptance.
Acceptance fraction is no longer fixed at 30%.
Logs include mean, std, bounded candidate, smoothed tau and observed scan acceptance.

Source/target classification losses retain separate means:
Lcls = Lsource + lambda_target * Ltarget.
lambda_target = 0.1, 0.2, 0.3, 0.4, 0.5 for the first five target epochs, then 0.5.
Source radius, rescue, competition, stable promotion and other losses retain base behavior.
Default LR is 0.001; workers default to 10.
Default run_name is global_mean_std_ema_v26.
threshold_quantile, threshold_beta and threshold_margin are legacy unused CLI options.

Run with your source checkpoint and --pre_epochs 0. Relevant options:
--threshold_std_scale 0.5 --threshold_smoothing 0.8 --threshold_min 0.85 --threshold_max 0.99

Tests:
python -m unittest discover -s tests -p test_global_mean_std_ema.py -v
Tests execute actual function bodies with a NumPy tensor shim; full PyTorch training
has not been verified in the authoring environment.
