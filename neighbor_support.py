"""Equal-weight source-neighbor support; no cumulative evidence amplification."""
import numpy as np
from dual_domain_ds import EvidenceBank


def otsu_threshold(scores):
    """Exact Otsu split over distinct scores; degenerate input rejects all.

    Return the smallest score in the high group, for inclusive >= selection.
    """
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    levels, counts = np.unique(values, return_counts=True)
    if len(levels) < 2:
        return float('inf')
    n = counts.sum()
    left_n = np.cumsum(counts)[:-1]
    sums = np.cumsum(levels * counts)
    left_mu = sums[:-1] / left_n
    right_mu = (sums[-1] - sums[:-1]) / (n - left_n)
    variance = left_n * (n - left_n) / float(n * n) * (left_mu - right_mu) ** 2
    return float(levels[np.argmax(variance) + 1])


class MeanSupportBank(EvidenceBank):
    def evaluate(self, queries, query_ids, config, exclude_self=False):
        z = np.asarray(queries, dtype=np.float32)
        ids = np.asarray(query_ids, dtype=np.int64)
        n = len(z)
        if z.ndim != 2 or z.shape[1] != self.x.shape[1] or ids.shape != (n,):
            raise ValueError('invalid query shapes')
        result = dict(mass=np.zeros((n, self.classes)), uncertainty=np.ones(n),
                      conflict=np.zeros(n), count=np.zeros(n, dtype=np.int64),
                      available=np.zeros(n, dtype=bool))
        norm = np.linalg.norm(z, axis=1)
        valid = np.isfinite(z).all(1) & (norm > 1e-8)
        z = np.nan_to_num(z) / np.maximum(np.nan_to_num(norm[:, None]), 1e-8)
        for start in range(0, n, 32):
            similarities = z[start:start + 32] @ self.x.T
            for offset, similarity in enumerate(similarities):
                i = start + offset
                if not valid[i]:
                    continue
                allowed = np.ones(len(self.x), dtype=bool)
                if exclude_self:
                    allowed &= self.ids != ids[i]
                candidates = np.flatnonzero(allowed)
                selected = candidates[np.argsort(-similarity[candidates], kind='stable')[:config.k]]
                result['count'][i] = len(selected)
                if len(selected) < config.min_neighbors:
                    continue
                strength = np.clip(similarity[selected], 0, 1)
                mass = np.bincount(self.labels[selected], weights=strength,
                                   minlength=self.classes) / len(selected)
                result['mass'][i] = mass
                result['uncertainty'][i] = max(0., 1. - mass.sum())
                result['available'][i] = True
        return result


def audit_mean_support(source, pseudo_labels, threshold):
    mass = source['mass']
    labels = np.asarray(pseudo_labels, dtype=np.int64)
    if labels.shape != (len(mass),) or ((labels < 0) | (labels >= mass.shape[1])).any():
        raise ValueError('invalid pseudo-labels')
    support = mass[np.arange(len(labels)), labels]
    competitors = mass.copy()
    competitors[np.arange(len(labels)), labels] = -np.inf
    return dict(pass_mask=source['available'] & (support >= threshold),
                available=source['available'], support=support,
                margin=support - competitors.max(1),
                uncertainty=source['uncertainty'], mass=mass)
