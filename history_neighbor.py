"""Experimental history-group exclusion gate. No target ground-truth labels.

CPU NumPy bank; epoch snapshots prevent same-epoch rescue feedback.
Exact history signatures are correlation proxies, not independent evidence.
"""
import numpy as np


class HistoryNeighborGate:
    def __init__(self, num_samples, feature_dim, window=4, k=20,
                 support=0.8, min_remaining=3, min_groups=2, min_similarity=0.5):
        if min(num_samples, feature_dim, window, k, min_remaining, min_groups) < 1:
            raise ValueError('sizes must be positive')
        if not 0 <= support <= 1 or not -1 <= min_similarity <= 1:
            raise ValueError('invalid support/similarity')
        self.window, self.k = window, k
        self.support, self.min_remaining = support, min_remaining
        self.min_groups, self.min_similarity = min_groups, min_similarity
        self.features = np.zeros((num_samples, feature_dim), np.float32)
        self.labels = np.full((num_samples, window), -1, np.int64)
        self.accepted = np.zeros((num_samples, window), bool)
        self.anchors = np.zeros((num_samples, window), bool)
        self.confidence = np.full((num_samples, window), np.nan, np.float32)
        self.seen = np.zeros((num_samples, window), bool)
        self.epoch = -1

    def begin_epoch(self, epoch):
        if epoch != self.epoch + 1:
            raise ValueError('epochs must be consecutive')
        # Query evidence contains only completed epochs.
        self.snapshot = {name: getattr(self, name).copy() for name in
                         ('features', 'labels', 'accepted', 'anchors', 'confidence', 'seen')}
        for name in ('labels', 'accepted', 'anchors', 'confidence', 'seen'):
            arr = getattr(self, name)
            arr[:, :-1] = arr[:, 1:]
            arr[:, -1] = -1 if name == 'labels' else np.nan if name == 'confidence' else False
        self.epoch = epoch

    @staticmethod
    def signature(labels, accepted, anchors, confidence):
        # Confidence-change bins: falling / approximately flat / rising.
        delta = np.diff(confidence)
        bins = np.where(delta < -0.05, -1, np.where(delta > 0.05, 1, 0))
        return tuple(np.concatenate((labels, accepted.astype(int), anchors.astype(int), bins)).tolist())

    def query(self, ids, features, labels, return_details=False):
        ids, features, labels = np.asarray(ids), np.asarray(features), np.asarray(labels)
        snap = self.snapshot
        n = len(ids)
        ok = np.zeros(n, bool)
        scores = np.zeros(n, np.float32)
        remaining = np.zeros(n, np.int64)
        group_counts = np.zeros(n, np.int64)
        details = {key: np.full(n, np.nan, np.float32) for key in
                   ('support_before', 'support_after', 'local_margin', 'nearest_similarity',
                    'mean_similarity', 'largest_group_fraction', 'groups_before', 'neighbors_before')}
        bank_ids = np.flatnonzero(snap['seen'].all(1) & snap['anchors'][:, -1])
        for row, sid in enumerate(ids):
            if not snap['seen'][sid].all() or not np.isfinite(features[row]).all():
                continue
            candidates = bank_ids[bank_ids != sid]
            if not len(candidates):
                continue
            z = features[row] / max(np.linalg.norm(features[row]), 1e-12)
            sims = snap['features'][candidates] @ z
            valid = sims >= self.min_similarity
            candidates, sims = candidates[valid], sims[valid]
            order = np.argsort(-sims, kind='stable')[:self.k]
            candidates, sims = candidates[order], sims[order]
            groups = {}
            for bid, sim in zip(candidates, sims):
                sig = self.signature(*(snap[key][bid] for key in
                                      ('labels', 'accepted', 'anchors', 'confidence')))
                groups.setdefault(sig, []).append((int(bid), float(sim)))
            if not groups:
                continue
            neighbor_labels = snap['labels'][candidates, -1]
            details['support_before'][row] = np.mean(neighbor_labels == labels[row])
            label_scores = {int(c): float(sims[neighbor_labels == c].mean())
                            for c in np.unique(neighbor_labels)}
            assigned = label_scores.get(int(labels[row]))
            competitors = [v for c, v in label_scores.items() if c != labels[row]]
            if assigned is not None and competitors:
                details['local_margin'][row] = assigned - max(competitors)
            details['nearest_similarity'][row] = sims.max()
            details['mean_similarity'][row] = sims.mean()
            details['largest_group_fraction'][row] = max(map(len, groups.values())) / len(candidates)
            details['groups_before'][row] = len(groups)
            details['neighbors_before'][row] = len(candidates)
            query_sig = self.signature(*(snap[key][sid] for key in
                                        ('labels', 'accepted', 'anchors', 'confidence')))
            # Remove the closest history group, even when no exact match exists.
            excluded = min(groups, key=lambda sig: sum(a != b for a, b in zip(sig, query_sig)))
            del groups[excluded]
            remaining[row] = sum(len(g) for g in groups.values())
            group_counts[row] = len(groups)
            votes = [np.mean([snap['labels'][bid, -1] == labels[row] for bid, _ in g])
                     for g in groups.values()]
            if votes:
                details['support_after'][row] = np.mean(votes)
            if remaining[row] < self.min_remaining or len(groups) < self.min_groups:
                continue
            # One contribution per history group; duplicates do not increase its weight.
            votes = [np.mean([snap['labels'][bid, -1] == labels[row] for bid, _ in g])
                     for g in groups.values()]
            scores[row] = np.mean(votes)
            ok[row] = scores[row] >= self.support
        result = (ok, scores, remaining, group_counts)
        return (*result, details) if return_details else result

    def observe(self, ids, features, labels, accepted, anchors, confidence):
        ids = np.asarray(ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids) or self.seen[ids, -1].any():
            raise ValueError('observe each sample once per epoch')
        features = np.asarray(features, dtype=np.float32)
        self.features[ids] = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        for key, values in (('labels', labels), ('accepted', accepted),
                            ('anchors', anchors), ('confidence', confidence)):
            getattr(self, key)[ids, -1] = values
        self.seen[ids, -1] = True

    def state_dict(self):
        return {'epoch': self.epoch, **{key: getattr(self, key).copy() for key in
                ('features', 'labels', 'accepted', 'anchors', 'confidence', 'seen')}}
