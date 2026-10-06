"""Deterministic class-wise spherical k-means for source-only prototypes."""
import numpy as np


def class_multi_prototypes(features, labels, k=3, iterations=20):
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y) or k < 1 or iterations < 1:
        raise ValueError('invalid feature/label shapes or clustering settings')
    if not np.isfinite(x).all() or np.any((y < 0) | (y >= 7)):
        raise ValueError('invalid features or class IDs')
    def normalize(a):
        return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    x = normalize(x)
    result, counts = [], []
    for c in range(7):
        samples = x[y == c]
        if len(samples) < k:
            raise ValueError('each source class must have at least k training samples')
        mean = normalize(samples.mean(0))
        # k=1 reproduces B: normalize the mean of unit source features.
        if k == 1:
            result.append(mean[None]); counts.append([len(samples)])
            continue
        # First member nearest class mean; successive farthest members.
        first = int(np.argmax(samples @ mean))
        selected = [first]
        centers = [samples[first]]
        for _ in range(1, k):
            similarity = (samples @ np.stack(centers).T).max(1)
            similarity[selected] = np.inf
            idx = int(np.argmin(similarity))
            selected.append(idx); centers.append(samples[idx])
        centers = np.stack(centers)
        for _ in range(iterations):
            similarities = samples @ centers.T
            assignment = similarities.argmax(1)
            updated = centers.copy()
            empty_used = set()
            for j in range(k):
                members = samples[assignment == j]
                if len(members):
                    candidate = normalize(members.mean(0))
                    if np.linalg.norm(candidate) > 1e-8:
                        updated[j] = candidate
                else:
                    distance = similarities.max(1).copy()
                    for idx in empty_used:
                        distance[idx] = np.inf
                    idx = int(distance.argmin())
                    empty_used.add(idx); updated[j] = samples[idx]
            converged = np.allclose(updated, centers, atol=1e-6)
            centers = updated
            if converged:
                break
        assignment = (samples @ centers.T).argmax(1)
        result.append(centers)
        counts.append(np.bincount(assignment, minlength=k).tolist())
    return np.stack(result), counts
