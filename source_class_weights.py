"""Class weights derived exclusively from labeled source-training counts."""
import numpy as np


def source_class_weights(counts, mode='none', power=0.5, max_weight=2.0):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0 or not np.isfinite(counts).all() or (counts < 0).any() or not (counts > 0).any():
        raise ValueError('counts must be a nonempty finite nonnegative vector with positive total')
    if mode not in ('none', 'inverse_power'):
        raise ValueError('unknown source class weight mode')
    if not np.isfinite(power) or not 0 <= power <= 1:
        raise ValueError('power must be finite and in [0, 1]')
    if not np.isfinite(max_weight) or max_weight < 1:
        raise ValueError('max_weight must be finite and >= 1')
    if mode == 'none':
        return np.ones(counts.size, dtype=np.float32)
    present = counts > 0
    weights = np.zeros_like(counts)
    weights[present] = counts[present] ** (-power)
    weights[present] /= weights[present].mean()
    # Cap AFTER mean normalization; do not renormalize beyond the cap.
    return np.minimum(weights, max_weight).astype(np.float32)
