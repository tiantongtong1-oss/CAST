"""Epoch-wide class means of dual-view minimum confidence (no label input)."""
import numpy as np


class DualViewClassMeans:
    def __init__(self, classes):
        self.classes = int(classes)
        if self.classes < 2:
            raise ValueError('classes must be at least two')
        self.sums = np.zeros(self.classes, dtype=np.float64)
        self.counts = np.zeros(self.classes, dtype=np.int64)

    def update(self, probabilities1, probabilities2):
        p1 = np.asarray(probabilities1, dtype=np.float64)
        p2 = np.asarray(probabilities2, dtype=np.float64)
        if p1.ndim != 2 or p1.shape != p2.shape or p1.shape[1] != self.classes:
            raise ValueError('matching B x C probability arrays required')
        for p in (p1, p2):
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError('probabilities must be finite and in [0, 1]')
            if not np.allclose(p.sum(axis=1), 1., atol=1e-5):
                raise ValueError('probabilities must sum to one')
        pred1, pred2 = p1.argmax(axis=1), p2.argmax(axis=1)
        agree = pred1 == pred2
        scores = np.minimum(p1.max(axis=1), p2.max(axis=1))
        self.sums += np.bincount(pred1[agree], weights=scores[agree], minlength=self.classes)
        self.counts += np.bincount(pred1[agree], minlength=self.classes)

    def finalize(self):
        # Infinity explicitly closes the direct route for unobserved classes.
        thresholds = np.full(self.classes, np.inf, dtype=np.float64)
        np.divide(self.sums, self.counts, out=thresholds, where=self.counts > 0)
        return thresholds, self.counts.copy()
