"""Class-conditional kNN audit. Never returns a training acceptance mask."""
import numpy as np


def class_neighbor_scores(query, pseudo_labels, features, labels, k=10,
                          num_classes=7, query_ids=None, bank_ids=None):
    query = np.asarray(query, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)
    if k < 1 or num_classes < 2 or query.ndim != 2 or features.ndim != 2:
        raise ValueError('invalid dimensions or k')
    if query.shape[1] != features.shape[1] or labels.shape != (len(features),) or pseudo_labels.shape != (len(query),):
        raise ValueError('shape mismatch')
    if np.any((pseudo_labels < 0) | (pseudo_labels >= num_classes)):
        raise ValueError('invalid pseudo label')
    if (query_ids is None) != (bank_ids is None):
        raise ValueError('both query and bank IDs required for self exclusion')
    if query_ids is not None:
        query_ids, bank_ids = np.asarray(query_ids), np.asarray(bank_ids)
        if query_ids.shape != (len(query),) or bank_ids.shape != (len(features),):
            raise ValueError('ID shape mismatch')
        if len(np.unique(bank_ids)) != len(bank_ids):
            raise ValueError('bank IDs must be unique')
    valid = np.isfinite(features).all(1) & (np.linalg.norm(features, axis=1) > 1e-12)
    query_valid = np.isfinite(query).all(1) & (np.linalg.norm(query, axis=1) > 1e-12)
    q = np.nan_to_num(query) / np.maximum(np.linalg.norm(np.nan_to_num(query), axis=1, keepdims=True), 1e-12)
    f = np.nan_to_num(features) / np.maximum(np.linalg.norm(np.nan_to_num(features), axis=1, keepdims=True), 1e-12)
    scores = np.full((len(query), num_classes), np.nan, np.float32)
    counts = np.zeros((len(query), num_classes), np.int64)
    for c in range(num_classes):
        idx = np.flatnonzero(valid & (labels == c))
        if not len(idx):
            continue
        sims = np.clip(q @ f[idx].T, -1, 1)
        if query_ids is not None:
            sims[query_ids[:, None] == bank_ids[idx][None, :]] = -np.inf
        # Partition per class: never infer missing competitor from global top-k.
        counts[:, c] = np.isfinite(sims).sum(1)
        if len(idx) >= k:
            best = np.partition(sims, len(idx)-k, axis=1)[:, -k:]
            ready = query_valid & (counts[:, c] >= k)
            scores[ready, c] = best[ready].mean(1)
    assigned = scores[np.arange(len(query)), pseudo_labels]
    other = scores.copy(); other[np.arange(len(query)), pseudo_labels] = np.nan
    competitors = np.full(len(query), -1, np.int64)
    competitor_scores = np.full(len(query), np.nan, np.float32)
    for i in range(len(query)):
        if np.isfinite(other[i]).any():
            competitors[i] = int(np.nanargmax(other[i]))
            competitor_scores[i] = other[i, competitors[i]]
    # All classes required for comparable margins. Partial values remain diagnostic only.
    ready = np.isfinite(scores).all(1)
    margin = np.where(ready, assigned - competitor_scores, np.nan)
    return dict(scores=scores, counts=counts, ready=ready, assigned=assigned,
                competitor=competitors, competitor_score=competitor_scores, margin=margin)
