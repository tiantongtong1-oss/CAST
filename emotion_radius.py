"""Source-only, class-balanced radii. Grouping never changes seven-class labels."""
import math

POSITIVE_CLASSES = (0, 3, 6)  # surprise, happy, neutral (user-defined grouping)
NEGATIVE_CLASSES = (1, 2, 4, 5)


def balanced_quantile(class_distances, classes, q):
    if not 0 < q <= 1:
        raise ValueError('q must be in (0, 1]')
    rows = []
    for c in classes:
        values = list(class_distances.get(c, []))
        if not values:
            continue
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid source distances')
        rows.extend((float(v), 1.0 / len(values)) for v in values)
    if not rows:
        return None
    rows.sort(key=lambda row: row[0])
    total = math.fsum(w for _, w in rows)
    cumulative = 0.
    for distance, weight in rows:
        cumulative += weight
        if cumulative >= q * total - 1e-12 * total:
            return distance
    return rows[-1][0]


def grouped_radii(class_distances):
    return (
        balanced_quantile(class_distances, POSITIVE_CLASSES, .85),
        balanced_quantile(class_distances, NEGATIVE_CLASSES, .90),
    )
