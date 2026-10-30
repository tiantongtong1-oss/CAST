"""Choose one strict distance threshold using equally weighted source classes."""
import numpy as np


def select_discriminative_radius(positive_by_class, negative_by_class):
    """Maximize mean_c P(d+ < R|c) - mean_c P(d- < R|c).

    d- is the distance to the nearest initialized OTHER class prototype.
    Search all empirical change points, group ties, and prefer the smallest
    radius on equal scores. R=0 is included (reject all if no positive gain).
    Float32 nextafter makes the selected boundary compatible with the bank's
    float32 buffer and its strict '<' gate. No target labels are consumed.
    """
    if len(positive_by_class) != len(negative_by_class):
        raise ValueError('positive and negative class lists must match')
    positives = [np.asarray(a, dtype=np.float32).reshape(-1) for a in positive_by_class]
    negatives = [np.asarray(a, dtype=np.float32).reshape(-1) for a in negative_by_class]
    for p, n in zip(positives, negatives):
        if p.shape != n.shape or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('positive distances must be finite, nonnegative and paired')
        if np.isnan(n).any() or (n < 0).any():
            raise ValueError('negative distances must be nonnegative; +inf means no competitor')
    active = sum(bool(p.size) for p in positives)
    radius = 0.0
    if active:
        distances, changes = [], []
        for p, n in zip(positives, negatives):
            if not p.size:
                continue
            weight = np.longdouble(1) / (active * p.size)
            finite_n = n[np.isfinite(n)]
            distances.extend([p, finite_n])
            changes.extend([np.full(p.size, weight, dtype=np.longdouble),
                            np.full(finite_n.size, -weight, dtype=np.longdouble)])
        distances, changes = np.concatenate(distances), np.concatenate(changes)
        order = np.argsort(distances, kind='stable')
        ordered = distances[order]
        ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(ordered) - 1]
        scores = np.cumsum(changes[order], dtype=np.longdouble)[ends]
        best = scores.max()
        tolerance = 1e-12  # Ignore floating accumulation noise when breaking ties.
        if best > tolerance:
            index = np.flatnonzero(scores >= best - tolerance)[0]
            boundary = ordered[ends[index]]
            radius = float(np.nextafter(boundary, np.float32(np.inf)))
            if not np.isfinite(radius):
                raise ValueError('radius exceeds float32 range')
    coverage = [float((p < radius).mean()) if p.size else None for p in positives]
    intrusion = [float((n < radius).mean()) if n.size else None for n in negatives]
    mean_coverage = sum(x for x in coverage if x is not None) / active if active else 0.0
    mean_intrusion = sum(x for x in intrusion if x is not None) / active if active else 0.0
    return dict(radius=radius, class_coverage=coverage, class_intrusion=intrusion,
                coverage=mean_coverage, intrusion=mean_intrusion,
                score=mean_coverage - mean_intrusion)
