"""NumPy-only source-neighbor diagnostics. Never changes training decisions."""
import numpy as np
from dual_domain_ds import fuse_simple_support


def diagnose_query(bank, feature, pseudo_label, query_id, config, ks=(1, 3, 5, 11)):
    z = np.asarray(feature, dtype=np.float32)
    if z.shape != (bank.x.shape[1],) or not np.isfinite(z).all() or np.linalg.norm(z) <= 1e-8:
        raise ValueError('diagnostic query must be a finite nonzero feature vector')
    label = int(pseudo_label)
    if not 0 <= label < bank.classes:
        raise ValueError('invalid pseudo-label')
    z = z / np.linalg.norm(z)
    distance = np.sqrt(np.maximum(2 - 2 * (bank.x @ z), 0))
    allowed = np.flatnonzero(distance <= config.max_distance)
    max_k = max(config.k, *ks)
    selected = allowed[np.argsort(distance[allowed], kind='stable')[:max_k]]
    r = np.minimum(bank.quality[selected] * np.exp(-.5 * (distance[selected] / config.bandwidth)**2),
                   config.strength_cap)
    neighbors = [dict(sample_id=int(query_id), rank=i+1, source_id=int(bank.ids[j]),
                      neighbor_label=int(bank.labels[j]), distance=float(distance[j]),
                      quality=float(bank.quality[j]), strength=float(r[i]),
                      strength_capped=bool(r[i] >= config.strength_cap),
                      same_label=bool(bank.labels[j] == label)) for i, j in enumerate(selected)]
    same = bank.labels == label
    d_same = float(distance[same].min()) if same.any() else None
    d_other = float(distance[~same].min()) if (~same).any() else None
    gap = d_other - d_same if d_same is not None and d_other is not None else None
    summary = dict(d_same=d_same, d_other=d_other, distance_gap=gap)
    for k in sorted(set(ks) | {config.k}):
        idx, strength = selected[:k], r[:k]
        b, u, conflict = fuse_simple_support(bank.labels[idx], strength, bank.classes)
        competitors = np.delete(b, label)
        usable = len(idx) >= config.min_neighbors and conflict <= config.conflict and conflict < 1-1e-12
        summary.update({f'count_k{k}':len(idx), f'support_k{k}':float(b[label]),
                        f'unknown_k{k}':float(u), f'conflict_k{k}':float(conflict),
                        f'margin_k{k}':float(b[label]-competitors.max()),
                        f'available_k{k}':bool(usable)})
    idx, strength = selected[:config.k], r[:config.k]
    supports = bank.labels[idx] == label
    summary.update(same_fraction=float(supports.mean()) if len(idx) else None,
                   weighted_support=float((strength*supports).sum()/strength.sum()) if strength.sum() else None,
                   capped_fraction=float((strength >= config.strength_cap).mean()) if len(idx) else None)
    return summary, neighbors



def view_probability_metrics(probabilities1, probabilities2):
    """Per-query JS divergence (natural logs, NOT square-root JS distance).

    Returns values in [0, ln(2)] plus the maximum-confidence difference.
    Inputs must be nonnegative finite probability matrices with equal shapes.
    Normalize within a small softmax-roundoff tolerance, preserving exact zeros.
    """
    p = np.asarray(probabilities1, dtype=np.float64)
    q = np.asarray(probabilities2, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] < 2:
        raise ValueError('views must have matching [samples, classes] shapes')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise ValueError('invalid view probabilities')
    for a in (p, q):
        if not np.allclose(a.sum(1), 1., rtol=1e-5, atol=1e-7):
            raise ValueError('probabilities must sum to one')
    p = p / p.sum(1, keepdims=True)
    q = q / q.sum(1, keepdims=True)
    m = .5 * (p+q)
    def kl(a):
        terms = np.zeros_like(a)
        positive = a > 0
        terms[positive] = a[positive] * (np.log(a[positive])-np.log(m[positive]))
        return terms.sum(1)
    js = np.clip(.5 * (kl(p)+kl(q)), 0., np.log(2.))
    return js, np.abs(p.max(1)-q.max(1))

