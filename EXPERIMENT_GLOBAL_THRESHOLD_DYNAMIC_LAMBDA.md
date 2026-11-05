# Shared target threshold with dynamic lambda

Parent: experiment/raw-medoid-q90-original-aug-lambda.

Each target epoch, the EMA teacher scans ALL target training images using deterministic original preprocessing plus a randomly sampled weak augmentation. For each image compute s=min(max_softmax(original),max_softmax(augmented)). Include both agreeing and disagreeing predictions in this scan. The global mean is sample-weighted across all batches, never an average of class means or batch means. Seven classes share exactly one threshold tau=lambda(epoch)*mean(s).

Admission still requires original/augmented class agreement and both confidences >= the same tau. New augmentations are sampled during training: the scan mean estimates that augmentation distribution, not the identical augmented tensors later used in training.

Dynamic lambda defaults to linear progress from 0.9 at epoch 0 to 1.0 at the last target epoch (30 epochs: lambda=0.9+0.1*epoch/29). --lambda_start/--lambda_end configure endpoints; --lambda_ramp_epochs configures the ramp length, default total target epochs. After the ramp lambda stays at its end value. A one-epoch ramp uses the end value. This is a deterministic curriculum, NOT optimal lambda estimation or pseudo-label accuracy calibration. Actual tau need not increase monotonically because the confidence mean also changes.

--threshold_lambda VALUE explicitly enables the previous fixed-lambda mode and overrides the schedule. Do NOT include this argument when requesting dynamic lambda. There is no clipping: tau is exactly lambda*mean. Default lambda<=1 keeps tau<=1; custom endpoints above1 may yield tau>1.

The radius gate L2-normalizes source features before recomputing the global medoid and Q90 radius, and L2-normalizes target original-view features before checking d<R (strict inequality, scale 1.0). Raw source/target features remain unchanged for kNN; classifier inputs are unchanged. Per-epoch source refresh and OR neighbor rescue remain in place. High-confidence admission works from epoch0; radius/KNN rescue starts at epoch3 by default. Dynamic lambda controls only the confidence path; rescue can accept samples below tau.

Log prefix: logs/globaltau_dynlambda_TIMESTAMP_PID.log. Per epoch logs Global_Min_Confidence_Mean, Lambda, Global_Threshold and Threshold_Sample_Count. Checkpoints use _globaltau_dynlambda_normmedoid_q90_r_or_ suffix.

Validation: three stdlib unit tests pass for schedule endpoints, override, invalid configuration and sample-weighted statistics; Python syntax/static integration checks pass. PyTorch unavailable in authoring environment; no tensor/GPU training validation performed.


## Cross-epoch threshold EMA (momentum 0.8)

The seven classes still share one threshold. At epoch e compute raw_tau[e]=lambda[e]*mean_i(min(conf_original[i],conf_augmented[i])) over ALL target training samples. Initialize tau[0]=raw_tau[0], then tau[e]=0.8*tau[e-1]+0.2*raw_tau[e]. The previous SMOOTHED threshold is used, not the previous raw estimate. Every batch in that epoch uses the same smoothed scalar. Both fixed and dynamic lambda modes use this smoothing. No additional clipping. This supersedes the direct per-epoch replacement described above. Checkpoint loading remains weights initialization, so a new run initializes threshold history from its first scan.

Logs include Raw_Global_Threshold, EMA_Momentum and Global_Threshold (the actual smoothed threshold used). Six threshold/schedule unit tests and syntax/static integration checks pass; full GPU training not run in the authoring environment.


## Normalized radius gate

Only the radius path normalizes features. The source raw memory is retained for Euclidean kNN. A separate L2-normalized source copy determines the exact medoid and linearly interpolated Q90 radius each epoch; target queries are normalized with the same operation before testing distance < R. There is no 0.7 multiplier. The historical RawMedoidQuantileBank class name is retained for import compatibility. Zero/nonfinite features and invalid labels are rejected; fewer than two valid source rows disables the radius gate. Dynamic lambda, threshold EMA, classifier inputs and prototype calculations are unchanged.

Validation for this change: Python syntax compilation, static radius-path isolation checks and stdlib numerical fixture checks passed. Regression tensor tests cover normalized medoid/Q90, source/target magnitude invariance, full-R admission, invalid features, insufficient source memory and unchanged raw kNN. These tensor tests and GPU training were not run because PyTorch is unavailable in the editing environment.
