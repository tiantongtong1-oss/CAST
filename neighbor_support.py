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


def select_support_threshold(scores, mode='otsu', fixed=0.92, min_count=128,
                             min_spread=0.01, min_separation=0.7,
                             min_group_fraction=0.05):
    """Select a threshold; guard defaults are experimental, not calibrated accuracy."""
    if mode not in ('otsu', 'fixed', 'otsu_guarded'):
        raise ValueError('invalid support threshold mode')
    if not np.isfinite(fixed) or not 0 <= fixed <= 1:
        raise ValueError('fixed support must be in [0, 1]')
    if min_count < 2 or not np.isfinite(min_spread) or min_spread < 0:
        raise ValueError('invalid count/spread guard')
    if not 0 <= min_separation <= 1 or not 0 <= min_group_fraction <= .5:
        raise ValueError('invalid separation/group guard')
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    raw = otsu_threshold(values)
    quantiles = np.quantile(values, [.05, .5, .95]) if len(values) else [np.nan] * 3
    separation = fraction = 0.0
    if np.isfinite(raw):
        high = values >= raw
        w = high.mean()
        fraction = min(w, 1 - w)
        total = values.var()
        if total > 0:
            separation = float(np.clip(w * (1-w) *
                (values[high].mean() - values[~high].mean()) ** 2 / total, 0, 1))
    reason = 'none'
    if mode == 'otsu_guarded':
        if not len(values):
            reason = 'no_candidates'
        elif len(values) < min_count:
            reason = 'too_few_candidates'
        elif not np.isfinite(raw):
            reason = 'constant_scores'
        elif quantiles[2] - quantiles[0] < min_spread:
            reason = 'narrow_distribution'
        elif separation < min_separation:
            reason = 'weak_separation'
        elif fraction < min_group_fraction:
            reason = 'small_group'
    effective_mode = ('fixed' if mode == 'fixed' else
                      'fixed_fallback' if reason != 'none' else 'otsu')
    threshold = raw if effective_mode == 'otsu' else fixed
    return dict(threshold=threshold, raw_otsu=raw, count=len(values),
                p05=float(quantiles[0]), p50=float(quantiles[1]), p95=float(quantiles[2]),
                separation=separation, min_group_fraction=fraction,
                mode=effective_mode, reason=reason)


class MeanSupportBank(EvidenceBank):
    def evaluate(self, queries, query_ids, config, exclude_self=False):
        # 查找源域邻居，计算各类别平均支持度。
        z = np.asarray(queries, dtype=np.float32)
        ids = np.asarray(query_ids, dtype=np.int64)
        n = len(z)
        if z.ndim != 2 or z.shape[1] != self.x.shape[1] or ids.shape != (n,):
            raise ValueError('invalid query shapes')
        result = dict(mass=np.zeros((n, self.classes)), uncertainty=np.ones(n),
                      conflict=np.zeros(n), count=np.zeros(n, dtype=np.int64),
                      available=np.zeros(n, dtype=bool))
        # 归一化
        norm = np.linalg.norm(z, axis=1)
        valid = np.isfinite(z).all(1) & (norm > 1e-8)
        z = np.nan_to_num(z) / np.maximum(np.nan_to_num(norm[:, None]), 1e-8)
        for start in range(0, n, 32):
            # 计算目标样本与所有源域样本的相似度，两边特征都归一化后，点积就是余弦相似度。
            similarities = z[start:start + 32] @ self.x.T
            for offset, similarity in enumerate(similarities):
                i = start + offset
                if not valid[i]:
                    continue
                # 按相似度从大到小，取前 K 个
                allowed = np.ones(len(self.x), dtype=bool)
                if exclude_self:
                    allowed &= self.ids != ids[i]
                candidates = np.flatnonzero(allowed)
                selected = candidates[np.argsort(-similarity[candidates], kind='stable')[:config.k]]
                result['count'][i] = len(selected)
                # 邻居不足就不审核
                if len(selected) < config.min_neighbors:
                    continue
                # 根据邻居类别计算支持度
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

