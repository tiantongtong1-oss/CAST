"""DS simple-support neighborhood audit; no target ground-truth input.

Singletons + Omega only. Domain-local Dempster fusion uses unnormalized
conjunctive masses to retain total conflict; cross-domain fusion is a linear pool.
NumPy CPU snapshots avoid retaining full reference sets on GPU.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DSConfig:
    k: int = 20
    min_neighbors: int = 3
    max_distance: float = 0.5
    bandwidth: float = 0.25
    alpha: float = 0.5
    support: float = 0.6
    margin: float = 0.1
    uncertainty: float = 0.3
    conflict: float = 0.5
    strength_cap: float = 0.95

    def __post_init__(self):
        if not 1 <= self.min_neighbors <= self.k:
            raise ValueError('require 1 <= ds_min_neighbors <= ds_k')
        for name in ('max_distance', 'bandwidth'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('DS distance and bandwidth must be finite and positive')
        for name in ('alpha', 'support', 'margin', 'uncertainty', 'conflict', 'strength_cap'):
            v = getattr(self, name)
            if not np.isfinite(v) or not 0 <= v <= 1:
                raise ValueError('DS coefficients/thresholds must be finite in [0,1]')
        if not 0 < self.strength_cap < 1:
            raise ValueError('ds_strength_cap must be strictly between zero and one')


def fuse_simple_support(labels, strengths, classes):
    """Return normalized singleton masses, Omega mass, total pre-normalization K.

    Empty evidence is vacuous. Totally conflicting evidence is returned vacuous
    with K=1, and must be rejected by the caller.
    """
    labels = np.asarray(labels, dtype=np.int64)
    strengths = np.asarray(strengths, dtype=np.float64)
    if labels.shape != strengths.shape or labels.ndim != 1:
        raise ValueError('evidence shapes must match')
    if ((labels < 0) | (labels >= classes)).any() or not np.isfinite(strengths).all() or ((strengths < 0) | (strengths > 1)).any():
        raise ValueError('invalid evidence')
    b, u = np.zeros(classes, dtype=np.float64), 1.0
    for label, r in zip(labels, strengths):
        old_b = b.copy()
        b *= 1 - r
        b[label] += (old_b[label] + u) * r
        u *= 1 - r
    survival = float(b.sum() + u)
    conflict = float(np.clip(1 - survival, 0, 1))
    if survival <= 1e-12:
        return np.zeros(classes), 1.0, conflict
    return b / survival, u / survival, conflict


class EvidenceBank:
    def __init__(self, features, labels, ids, quality, classes):
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)
        q = np.asarray(quality, dtype=np.float64)
        if x.ndim != 2 or any(a.shape != (len(x),) for a in (y, ids, q)):
            raise ValueError('invalid reference shapes')
        if len(np.unique(ids)) != len(ids):
            raise ValueError('reference IDs must be unique')
        valid = np.isfinite(x).all(1) & (np.linalg.norm(x, axis=1) > 1e-8)
        valid &= (y >= 0) & (y < classes) & np.isfinite(q) & (q > 0) & (q <= 1)
        x, y, ids, q = x[valid], y[valid], ids[valid], q[valid]
        self.x = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)).copy()
        self.labels, self.ids, self.quality = y.copy(), ids.copy(), q.copy()
        self.classes = classes

    def evaluate(self, queries, query_ids, config, exclude_self=False):
        z = np.asarray(queries, dtype=np.float32)
        ids = np.asarray(query_ids, dtype=np.int64)
        n = len(z)
        if z.ndim != 2 or z.shape[1] != self.x.shape[1] or ids.shape != (n,):
            raise ValueError('invalid query shapes')
        result = dict(mass=np.zeros((n, self.classes)), uncertainty=np.ones(n),
                      conflict=np.zeros(n), count=np.zeros(n, dtype=np.int64),
                      available=np.zeros(n, dtype=bool))
        valid = np.isfinite(z).all(1) & (np.linalg.norm(z, axis=1) > 1e-8)
        z = np.nan_to_num(z) / np.maximum(np.linalg.norm(np.nan_to_num(z), axis=1, keepdims=True), 1e-8)
        if not len(self.x):
            return result
        # Chunk queries to bound the B x reference-count distance allocation.
        for start in range(0, n, 32):
            dist = np.sqrt(np.maximum(2 - 2 * (z[start:start + 32] @ self.x.T), 0))
            for offset, distances in enumerate(dist):
                i = start + offset
                if not valid[i]:
                    continue
                allowed = distances <= config.max_distance
                if exclude_self:
                    allowed &= self.ids != ids[i]
                candidates = np.flatnonzero(allowed)
                # Stable sort ensures reproducible tie handling.
                selected = candidates[np.argsort(distances[candidates], kind='stable')[:config.k]]
                result['count'][i] = len(selected)
                if len(selected) < config.min_neighbors:
                    continue
                r = self.quality[selected] * np.exp(-0.5 * (distances[selected] / config.bandwidth) ** 2)
                r = np.minimum(r, config.strength_cap)
                b, u, k = fuse_simple_support(self.labels[selected], r, self.classes)
                result['conflict'][i] = k
                if k > config.conflict or k >= 1 - 1e-12:
                    continue  # retain vacuous mass for discounted/unavailable domain
                result['mass'][i], result['uncertainty'][i] = b, u
                result['available'][i] = True
        return result


def combine_domains(source, target, pseudo_labels, config):
    """Linear pool after domain DS; strict two-domain availability gate."""
    b_s, b_t = source['mass'], target['mass']
    labels = np.asarray(pseudo_labels, dtype=np.int64)
    if b_s.shape != b_t.shape or labels.shape != (len(b_s),):
        raise ValueError('domain and pseudo-label shapes must match')
    if ((labels < 0) | (labels >= b_s.shape[1])).any():
        raise ValueError('invalid pseudo-label')
    mass = config.alpha * b_s + (1 - config.alpha) * b_t
    uncertainty = config.alpha * source['uncertainty'] + (1 - config.alpha) * target['uncertainty']
    cross = np.maximum(b_s.sum(1) * b_t.sum(1) - (b_s * b_t).sum(1), 0)
    support = mass[np.arange(len(labels)), labels]
    competitors = mass.copy()
    competitors[np.arange(len(labels)), labels] = -np.inf
    margin = support - competitors.max(1)
    available = source['available'] & target['available']
    passed = available & (support >= config.support) & (margin >= config.margin)
    passed &= (uncertainty <= config.uncertainty) & (cross <= config.conflict)
    return dict(pass_mask=passed, available=available, support=support, margin=margin,
                uncertainty=uncertainty, cross_conflict=cross, mass=mass)
