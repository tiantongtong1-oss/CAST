"""Class-balanced shared diagonal Mahalanobis metric, calibrated on source only."""
import numpy as np
from balanced_radius import balanced_radius_quantile


def fit_shared_diagonal(class_features, quantile=0.90, regularizer=1e-4):
    """Inputs are valid, already unit-normalized N_c x D source features.

    Returns distinct class means, one shared within-class variance vector,
    one balanced quantile threshold, and per-class calibration distances.
    Empty classes have zero means and do not contribute to variance/quantile.
    """
    if isinstance(regularizer, (bool, np.bool_)) or not np.isfinite(regularizer) or regularizer <= 0:
        raise ValueError('regularizer must be finite and positive')
    arrays = [np.asarray(x, dtype=np.float64) for x in class_features]
    if not arrays or any(x.ndim != 2 for x in arrays):
        raise ValueError('nonempty list of N_c x D matrices required')
    dim = arrays[0].shape[1]
    if dim < 1 or any(x.shape[1] != dim or not np.isfinite(x).all() for x in arrays):
        raise ValueError('features must be finite and share a positive dimension')
    means = np.stack([x.mean(0) if len(x) else np.zeros(dim) for x in arrays])
    active = [((x - mu) ** 2).mean(0) for x, mu in zip(arrays, means) if len(x)]
    variance = np.mean(active, axis=0) if active else np.zeros(dim)
    distances = [np.sqrt(((x - mu) ** 2 / (variance + regularizer)).sum(1))
                 for x, mu in zip(arrays, means)]
    radius = balanced_radius_quantile(distances, quantile)
    return means, variance, radius, distances
