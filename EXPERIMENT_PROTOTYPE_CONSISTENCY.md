# Experiment: Prototype Consistency v3 + Confidence/Energy OR Rescue

Base experiment: `experiment/prototype-consistency-v2`

Experiment branch: `experiment/prototype-consistency-v3`

This version keeps the v2 EMA teacher, dual-view prediction agreement, class-adaptive confidence thresholds, source-anchored prototype consistency loss, and source class-distribution energy model. The key change is how confidence and energy are combined.

## Core change

v2 uses energy as a veto after confidence filtering:

```text
m_final = m_conf AND m_energy
```

v3 uses energy as a rescue signal for samples whose two weak views agree on the same class:

```text
m_agree = pred(view1) == pred(view2)

m_conf = m_agree
         AND confidence(view1) >= class threshold
         AND confidence(view2) >= class threshold

m_energy = m_agree
           AND source class distribution is initialized
           AND log E_pseudo(z) <= tau_energy,pseudo

m_rescue = m_agree AND m_energy AND NOT m_conf
m_final  = m_conf OR m_rescue
```

Equivalently:

```text
m_final = m_agree AND (m_high_confidence OR m_strong_energy_support)
```

The two weak views must still predict the same class. Energy does not rescue samples whose weak views disagree.

## Why the energy candidate set changed

In v2, the energy bank is evaluated only on samples that already pass the confidence filter. That is correct for a veto gate, but it cannot implement a real OR rule because `m_energy` is then a subset of `m_conf`.

v3 therefore evaluates energy on every sample satisfying `m_agree`. This allows an agreed target sample that misses the confidence threshold to be accepted when its representation is strongly supported by the predicted source class distribution.

## Stronger energy threshold for rescue

The energy threshold has a different role in v3. In v2, `energy_quantile=0.95` is a relatively permissive veto threshold: it mainly removes extreme geometric outliers.

In v3, energy alone may rescue a low-confidence pseudo label, so the default is deliberately stricter:

```text
energy_quantile = 0.75
```

For each class `c`, the threshold is still calibrated from source leave-one-out energies:

```text
tau_energy,c = quantile(log E_c(source), energy_quantile)
```

A lower quantile means a target sample must lie in a more strongly source-supported region before energy can rescue it.

## Safety for missing source distributions

`ClassDistributionBank.gate()` intentionally fails open when a class has no valid source distribution. That behavior is useful in v2 because it prevents a missing distribution from deleting all pseudo labels of that class.

For v3 rescue this would be unsafe: an uninitialized class must not rescue a low-confidence sample. Therefore v3 explicitly requires:

```text
source_initialized[pseudo_class] == True
```

before `m_energy` can contribute to rescue.

## Warmup

Default:

```text
energy_warmup_epochs = 1
```

During warmup, energy is computed and logged but cannot rescue samples:

```text
epoch < energy_warmup_epochs:
    m_final = m_conf
```

After warmup:

```text
m_final = m_conf OR m_rescue
```

## Downstream use

As in v2, the final mask is reused by the existing training pipeline. A rescued pseudo label therefore participates in:

- target classification loss;
- CAST target affinity alignment;
- prototype consistency loss;
- target EMA prototype updates.

No additional energy gradient loss is introduced.

## Distribution energy

The source class-distribution model is unchanged from v2. For normalized feature `z` and source class `c`:

```text
D_c(z) = (z - p_c)^T Sigma_c^{-1} (z - p_c)
```

Local source support is estimated with a Gaussian kernel in the same Mahalanobis geometry:

```text
L_c(z) = (1 / M_c) * sum_j exp(
    - (z - z_j)^T Sigma_c^{-1} (z - z_j) / (2 h^2)
)
```

The implementation compares:

```text
log E_c(z) = log(D_c(z) + eps) - log L_c(z)
```

rather than computing the ratio directly.

## Dynamic source-distribution rebuild

The source distribution bank is rebuilt using deterministic RAF-DB views and the current EMA teacher every `energy_refresh_interval` target epochs.

Default:

```text
energy_refresh_interval = 1
```

The fixed source prototype used by the prototype-consistency term remains unchanged.

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
energy_quantile             = 0.75
energy_cov_shrinkage        = 0.05
energy_max_density_samples  = 256
energy_warmup_epochs        = 1
energy_refresh_interval     = 1
```

## Diagnostics

Each target epoch reports:

```text
Agreement_Num
Confidence_Accept_Num
Energy_Pass_Num
Energy_Rescue_Num
Final_Accept_Num
Mean_Agreed_LogEnergy
Pseudo_Distribution_Confidence
Pseudo_Distribution_Rescued
Pseudo_Distribution_Final
```

`Energy_Rescue_Num` is the key v3 metric. It counts pseudo labels that would have been rejected by the strict confidence rule but were recovered by source-distribution support.

The per-class rescued distribution should be monitored carefully. If one class contributes a disproportionate number of rescued labels, reduce `energy_quantile` before changing other losses.

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
  --energy_quantile 0.75 \
  --energy_cov_shrinkage 0.05 \
  --energy_max_density_samples 256 \
  --energy_warmup_epochs 1 \
  --energy_refresh_interval 1 \
  2>&1 | tee logs/cast_prototype_consistency_v3_or_mobilenet_v2.log
```

## Suggested ablations

Keep all other settings fixed and compare:

```text
A. v2: confidence AND energy, quantile 0.95
B. v3 with --no_energy_gate (strict confidence only)
C. v3 OR rescue, quantile 0.75
D. v3 OR rescue, quantile 0.60
E. v3 OR rescue, quantile 0.85
```

Do not tune against the final test set. Use validation accuracy together with `Energy_Rescue_Num` and the per-class rescued distribution to judge whether the rescue condition is too strict or too permissive.
