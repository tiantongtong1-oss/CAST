# Experiment: Prototype Consistency v2 + Source Class-Distribution Energy Gate

Base experiment: `experiment/prototype-consistency-v1`

Experiment branch: `experiment/prototype-consistency-v2`

This version keeps the v1 EMA teacher, dual-view confidence filtering, class-adaptive thresholds, and source-anchored prototype consistency loss. It adds a second pseudo-label validation stage in representation space: a source class-distribution energy gate.

## Motivation

A target sample can pass the two-view confidence rule while still being geometrically atypical for its predicted source class. A single class prototype only describes the class center and does not capture anisotropic class spread or whether the sample lies in a locally supported region of the source distribution.

v2 therefore models each source class with:

- a class mean / prototype `p_c`;
- a shrinkage covariance `Sigma_c`;
- a Mahalanobis global deviation `D_c(z)`;
- a source-supported local density `L_c(z)`;
- a class-specific energy threshold calibrated only from labeled source features.

The energy gate is applied after the existing dual-view confidence rule. It does not use FER2013 training labels.

## Distribution energy

For class `c` and normalized representation `z`:

```text
D_c(z) = (z - p_c)^T Sigma_c^{-1} (z - p_c)
```

Local support is estimated from source representatives of class `c`:

```text
L_c(z) = (1 / M_c) * sum_j exp(
    - (z - z_j)^T Sigma_c^{-1} (z - z_j) / (2 h^2)
)
```

The conceptual energy is:

```text
E_c(z) = D_c(z) / (L_c(z) + eps)
```

The implementation compares `log E_c(z)` instead of `E_c(z)` directly:

```text
log E_c(z) = log(D_c(z) + eps) - log L_c(z)
```

This is monotonic with the original energy and avoids kernel underflow in the 512-D feature space. The density calculation uses `logsumexp`.

## Covariance stabilization

The source covariance is regularized before inversion:

```text
Sigma_reg = (1 - s) * Sigma
            + s * mean_variance * I
            + eps * I
```

The precision matrix is then obtained from a symmetric eigendecomposition rather than a direct matrix inverse. Default shrinkage is `s = 0.05`.

## Class-specific energy threshold

For each class, v2 computes leave-one-out source energies on the density representatives and sets:

```text
tau_energy,c = quantile(log E_c(source), energy_quantile)
```

Default:

```text
energy_quantile = 0.95
```

A target pseudo label `c` passes the distribution gate only when:

```text
log E_c(z_target) <= tau_energy,c
```

Classes without a valid source distribution fail open instead of deleting the entire pseudo class.

## Final pseudo-label mask

The existing confidence mask remains:

```text
m_conf = weak-view agreement
         AND confidence(view1) >= class threshold
         AND confidence(view2) >= class threshold
```

After the energy warmup:

```text
m_energy = log E_pseudo(z) <= tau_energy,pseudo
m_final  = m_conf AND m_energy
```

`m_final` is then reused by the existing training pipeline for target classification loss, CAST target affinity alignment, prototype consistency loss, and target EMA prototype updates.

The energy term is deliberately a gate, not an additional gradient loss, so v2 isolates pseudo-label reliability from representation regularization.

## Dynamic source-distribution rebuild

The fixed source prototype used by Prototype Consistency v1 is unchanged. The distribution bank is different: by default it is rebuilt at the beginning of every target epoch using deterministic RAF-DB views and the current EMA teacher.

This avoids comparing late-stage target features against covariance and density statistics measured in an obsolete early-stage teacher feature space.

Use `--energy_refresh_interval N` to rebuild less frequently if the extra source pass is too expensive.

## Density memory

Mean and covariance use all source features collected for each class. Local density uses at most `energy_max_density_samples` deterministic representatives per class to keep target-batch gating practical.

Default:

```text
energy_max_density_samples = 256
```

`energy_bandwidth` is dimension-normalized internally; the effective Gaussian bandwidth is:

```text
h_effective = energy_bandwidth * sqrt(feature_dim)
```

With the current 512-D feature heads, the default multiplier is `1.0`.

## Defaults

```text
proto_weight                = 0.10
proto_temperature           = 0.20
proto_momentum              = 0.99
proto_source_anchor         = 0.50
proto_warmup_epochs         = 3
proto_ramp_epochs           = 5

energy_gate                 = enabled
energy_bandwidth            = 1.0
energy_quantile             = 0.95
energy_cov_shrinkage        = 0.05
energy_max_density_samples  = 256
energy_warmup_epochs        = 1
energy_refresh_interval     = 1
```

## Diagnostics

Each target epoch logs the existing prototype diagnostics plus:

```text
Energy source counts
Class log-energy thresholds
Energy gate enforced
Confidence_Accept_Num
Energy_Pass_Num
Final_Accept_Num
Mean_Assigned_LogEnergy
Pseudo_Distribution_Before_Energy
Pseudo_Distribution_After_Energy
```

The before/after class histograms are important. If one minority class is almost eliminated by the energy gate, first relax `--energy_quantile` or increase `--energy_warmup_epochs` rather than adding more losses.

## Run

```bash
python train.py \
  --backbone mobilenet_v2 \
  --pre_epochs 30 \
  --epochs 30 \
  --lr 0.001 \
  --w1 4 \
  --w2 0.3 \
  --w3 0.1 \
  --ema_decay 0.999 \
  --threshold_base 0.85 \
  --threshold_beta 0.5 \
  --threshold_margin 0.02 \
  --threshold_min 0.80 \
  --threshold_max 0.95 \
  --proto_weight 0.10 \
  --proto_temperature 0.20 \
  --proto_momentum 0.99 \
  --proto_source_anchor 0.50 \
  --proto_warmup_epochs 3 \
  --proto_ramp_epochs 5 \
  --energy_gate \
  --energy_bandwidth 1.0 \
  --energy_quantile 0.95 \
  --energy_cov_shrinkage 0.05 \
  --energy_max_density_samples 256 \
  --energy_warmup_epochs 1 \
  --energy_refresh_interval 1 \
  2>&1 | tee logs/cast_prototype_consistency_v2_energy_mobilenet_v2.log
```

## Suggested ablations

Keep all other settings fixed and compare:

```text
A. v1: prototype consistency only
B. v2 with --no_energy_gate
C. v2 energy gate, quantile 0.95
D. v2 energy gate, quantile 0.97
```

If the gate rejects too many otherwise confident samples, first try a higher source energy quantile such as `0.97`; if it rejects almost nothing, try `0.90` or `0.925`. Do not tune against the final test set.
