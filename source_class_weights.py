"""Class weights derived only from source training labels."""
import numpy as np


def source_class_weights(counts, mode='none'):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not len(counts) or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError('source class counts must be finite and positive')
    if mode == 'none':
        return np.ones_like(counts)
    if mode != 'sqrt_inverse':
        raise ValueError('unknown source class weighting mode')
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    return weights / weights.mean()
