"""Class-balanced inverse empirical CDF for a shared source radius."""
import numpy as np


def balanced_radius_quantile(class_distances, quantile):
    """Return the smallest observed distance reaching the weighted quantile.

    Each nonempty class has equal total mass. Missing classes contribute no
    mass; an entirely empty source returns zero. Uses an inverse empirical CDF
    (no linear interpolation), unlike the old per-class torch.quantile rule.
    """
    if isinstance(quantile, (bool, np.bool_)) or not np.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError('quantile must be finite and in (0, 1]')
    classes = [np.asarray(values, dtype=np.float64).reshape(-1)
               for values in class_distances]
    if any(not np.isfinite(values).all() or (values < 0).any() for values in classes):
        raise ValueError('distances must be finite and nonnegative')
    active = [values for values in classes if values.size]
    if not active:
        return 0.0
    values = np.concatenate(active)
    if quantile == 1:
        return float(values.max())
    # Long-double accumulation reduces sensitivity to class size and ordering.
    weights = np.concatenate([
        np.full(values.size, np.longdouble(1) / (len(active) * values.size),
                dtype=np.longdouble) for values in active
    ])
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order], dtype=np.longdouble)
    # Absorb only roundoff at an exact empirical-CDF boundary.
    target = np.longdouble(quantile) * cumulative[-1]
    tolerance = 8 * np.finfo(np.float64).eps * cumulative[-1]
    index = np.searchsorted(cumulative, target - tolerance, side='left')
    return float(values[order[min(int(index), len(order) - 1)]])



def balanced_radius_mean_std(class_distances, std_scale=1.0):
    """Return (radius, mean, population std), giving each nonempty class equal mass."""
    if isinstance(std_scale, (bool, np.bool_)) or not np.isfinite(std_scale) or std_scale < 0:
        raise ValueError('std_scale must be finite and nonnegative')
    classes = [np.asarray(values, dtype=np.float64).reshape(-1)
               for values in class_distances]
    if any(not np.isfinite(values).all() or (values < 0).any() for values in classes):
        raise ValueError('distances must be finite and nonnegative')
    active = [values for values in classes if values.size]
    if not active:
        return 0.0, 0.0, 0.0
    # Center before squaring: includes both within-class and between-class variation.
    mean = np.mean([values.astype(np.longdouble).mean() for values in active])
    variance = np.mean([((values.astype(np.longdouble) - mean) ** 2).mean()
                        for values in active])
    std = np.sqrt(variance)
    result = np.asarray([mean + std_scale * std, mean, std], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('radius statistics must be finite')
    return tuple(float(value) for value in result)
