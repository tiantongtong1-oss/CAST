"""CPU source-only nearest-neighbor diagnostics; no training decisions."""
import numpy as np

NAMES = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']


class SourceKNNAudit:
    def __init__(self, features, labels, paths, k=20):
        features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        if features.ndim != 2 or len(features) != len(labels) or len(paths) != len(labels):
            raise ValueError('source features, labels and paths must align')
        if not np.isfinite(features).all() or np.any((self.labels < 0) | (self.labels > 6)):
            raise ValueError('invalid source features/labels')
        counts = np.bincount(self.labels, minlength=7)
        if counts.min() == 0:
            raise ValueError('balanced audit requires all seven source classes')
        self.features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        self.paths = paths
        self.k = min(k, len(labels))
        if self.k < 1:
            raise ValueError('k must be positive')
        rng = np.random.RandomState(2000)
        self.balanced_ids = np.concatenate([
            rng.choice(np.flatnonzero(self.labels == c), counts.min(), replace=False)
            for c in range(7)])
        self.counts = counts.tolist()
        self.per_class = int(counts.min())

    def query(self, features, predictions):
        features = np.asarray(features, dtype=np.float32)
        features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        if not np.isfinite(features).all():
            raise ValueError('invalid target features')
        rows, neighbors = [], []
        # Bound temporary CPU similarity matrices, independent of FER batch size.
        for start in range(0, len(features), 64):
            similarities = features[start:start + 64] @ self.features.T
            for j, scores in enumerate(similarities):
                row = {}
                raw_neighbors = None
                for prefix, pool in [('source', np.arange(len(self.labels))),
                                     ('balanced_source', self.balanced_ids)]:
                    k = min(self.k, len(pool))
                    ids = pool[np.argpartition(-scores[pool], k - 1)[:k]]
                    ids = ids[np.argsort(-scores[ids], kind='stable')]
                    votes = np.bincount(self.labels[ids], minlength=7) / k
                    prediction = int(votes.argmax())  # ties use lowest class ID
                    row.update({prefix + '_support_' + name: float(votes[c])
                                for c, name in enumerate(NAMES)})
                    row[prefix + '_knn_pred'] = prediction
                    row[prefix + '_knn_agrees'] = prediction == int(predictions[start + j])
                    row[prefix + '_mean_neighbor_cosine'] = float(scores[ids].mean())
                    if prefix == 'source':
                        raw_neighbors = [dict(rank=rank + 1, source_path=str(self.paths[idx]),
                                              source_true=int(self.labels[idx]),
                                              cosine=float(scores[idx]))
                                         for rank, idx in enumerate(ids)]
                rows.append(row)
                neighbors.append(raw_neighbors)
        return rows, neighbors
