# Shared target threshold with dynamic lambda

Parent: experiment/raw-medoid-q90-original-aug-lambda.

Each target epoch, the EMA teacher scans ALL target training images using deterministic original preprocessing plus a randomly sampled weak augmentation. For each image compute s=min(max_softmax(original),max_softmax(augmented)). Include both agreeing and disagreeing predictions in this scan. The global mean is sample-weighted across all batches, never an average of class means or batch means. Seven classes share exactly one threshold tau=lambda(epoch)*mean(s).

Admission still requires original/augmented class agreement and both confidences >= the same tau. New augmentations are sampled during training: the scan mean estimates that augmentation distribution, not the identical augmented tensors later used in training.

Dynamic lambda defaults to linear progress from 0.9 at epoch 0 to 1.0 at the last target epoch (30 epochs: lambda=0.9+0.1*epoch/29). --lambda_start/--lambda_end configure endpoints; --lambda_ramp_epochs configures the ramp length, default total target epochs. After the ramp lambda stays at its end value. A one-epoch ramp uses the end value. This is a deterministic curriculum, NOT optimal lambda estimation or pseudo-label accuracy calibration. Actual tau need not increase monotonically because the confidence mean also changes.

--threshold_lambda VALUE explicitly enables the previous fixed-lambda mode and overrides the schedule. Do NOT include this argument when requesting dynamic lambda. There is no clipping: tau is exactly lambda*mean. Default lambda<=1 keeps tau<=1; custom endpoints above1 may yield tau>1.

Inherited raw-feature medoid, source Q90 radius, d<0.7R gate, per-epoch source refresh, and OR neighbor rescue remain unchanged. High-confidence admission works from epoch0; radius/KNN rescue starts at epoch3 by default. Dynamic lambda controls only the confidence path; rescue can accept samples below tau.

Log prefix: logs/globaltau_dynlambda_TIMESTAMP_PID.log. Per epoch logs Global_Min_Confidence_Mean, Lambda, Global_Threshold and Threshold_Sample_Count. Checkpoints use _globaltau_dynlambda_rawmedoid_q90_07_or_ suffix.

Validation: three stdlib unit tests pass for schedule endpoints, override, invalid configuration and sample-weighted statistics; Python syntax/static integration checks pass. PyTorch unavailable in authoring environment; no tensor/GPU training validation performed.


## Cross-epoch threshold EMA (momentum 0.8)

The seven classes still share one threshold. At epoch e compute raw_tau[e]=lambda[e]*mean_i(min(conf_original[i],conf_augmented[i])) over ALL target training samples. Initialize tau[0]=raw_tau[0], then tau[e]=0.8*tau[e-1]+0.2*raw_tau[e]. The previous SMOOTHED threshold is used, not the previous raw estimate. Every batch in that epoch uses the same smoothed scalar. Both fixed and dynamic lambda modes use this smoothing. No additional clipping. This supersedes the direct per-epoch replacement described above. Checkpoint loading remains weights initialization, so a new run initializes threshold history from its first scan.

Logs include Raw_Global_Threshold, EMA_Momentum and Global_Threshold (the actual smoothed threshold used). Six threshold/schedule unit tests and syntax/static integration checks pass; full GPU training not run in the authoring environment.
