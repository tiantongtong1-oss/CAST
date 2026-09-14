# EMA Dual-View CAST innovation branch

Branch: `improve/ema-dualview-cast`

This branch keeps the RAF-DB -> FER2013 CAST baseline and adds the stability
and pseudo-label improvements shown in the proposed architecture diagram.
MobileNetV2 is the default backbone.

## Main changes

1. Dual-view EMA teacher
   - Two independent weak target views are evaluated by an EMA teacher.
   - Pseudo labels require view agreement and class-wise confidence checks.
   - Teacher parameters are updated by EMA and never receive gradients.

2. Global class-wise target statistics
   - Thresholds and the target class prior are estimated on the complete target
     training split at the start of each target epoch.
   - Mild inverse-prior distribution alignment corrects head/tail prediction bias.
   - A high-confidence fallback avoids an entirely empty target batch.

3. Improved classification loss
   - Source CE and target pseudo-label CE are normalized separately.
   - Target supervision is ramped in with `lambda_target`.
   - Target CE uses bounded confidence/class-balance sample weights.

4. DDRL stability
   - Existing MK-MMD class-conditional alignment and inter-class separation are
     preserved and logged separately.
   - Empty classes, NaN/Inf values, class-weight upper bounds and loss clipping
     are handled explicitly.

5. Prototype target affinity
   - EMA class prototypes are anchored by labeled source features.
   - Only reliable target pseudo labels update target prototypes.
   - Target features are pulled toward their class prototype while initialized
     prototypes are separated by a cosine-margin constraint.
   - Prototype affinity uses warm-up and ramp scheduling with `lambda_aff` as
     the upper bound.

6. Numerical safeguards
   - Gradient clipping, bounded pseudo-label weights, bounded class weights and
     finite-value guards are applied to the target adaptation stage.

7. Baseline protection
   - Improved checkpoints contain `_ema_dualview_` in their names so running
     this branch does not overwrite the baseline checkpoint files.

## Recommended MobileNetV2 run

```bash
python train.py \
  --backbone mobilenet_v2 \
  --pre_epochs 30 \
  --epochs 30 \
  --lr 0.001 \
  --w1 4 \
  --w2 0.3 \
  --w3 0.1 \
  --phi 1.4 \
  2>&1 | tee logs/cast_ema_dualview_mobilenet_v2.log
```

For a quick numerical sanity check, use `--pre_epochs 2 --epochs 2` first.
