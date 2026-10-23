"""Source-based competing-class audit; never imposes class quotas."""
import numpy as np


def competition_audit(prob1, prob2, features, labels, means, ready, votes,
                      available, probability_margin=0.1, prototype_margin=0.02,
                      vote_margin=3):
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    rows = np.arange(n)
    gaps = []
    for prob in (np.asarray(prob1), np.asarray(prob2)):
        other = prob.copy()
        other[rows, labels] = -np.inf
        gaps.append(prob[rows, labels] - other.max(axis=1))
    probability_gap = np.minimum(*gaps)
    features, means = np.asarray(features), np.asarray(means)
    distances = np.linalg.norm(features[:, None, :] - means[None, :, :], axis=2)
    valid_classes = np.asarray(ready, dtype=bool) & np.isfinite(means).all(axis=1)
    distances[:, ~valid_classes] = np.inf
    assigned = distances[rows, labels].copy()
    distances[rows, labels] = np.inf
    nearest_other = distances.min(axis=1)
    prototype_gap = nearest_other - assigned
    votes = np.asarray(votes)
    competing_votes = votes.copy()
    competing_votes[rows, labels] = -1
    vote_gap = votes[rows, labels] - competing_votes.max(axis=1)
    passed = np.isfinite(probability_gap) & (probability_gap >= probability_margin)
    passed &= valid_classes[labels] & np.isfinite(assigned) & np.isfinite(nearest_other)
    passed &= np.isfinite(features).all(axis=1) & (np.linalg.norm(features, axis=1) > 1e-8)
    # Match d(candidate) + delta < d(nearest competitor), strictly.
    passed &= prototype_gap > prototype_margin
    passed &= np.asarray(available, dtype=bool) & (vote_gap >= vote_margin)
    return dict(pass_mask=passed, probability_gap=probability_gap,
                prototype_gap=prototype_gap, vote_gap=vote_gap)


def supervision_mask(confidence, rescue, anchors):
    """High-confidence labels learn now; rescue labels learn only as anchors."""
    return confidence | (rescue & anchors)
