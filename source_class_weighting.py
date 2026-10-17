"""Source-label class weights and source-only CE reduction; no target statistics."""
import numpy as np


def make_source_class_weights(counts, power=0.5):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not len(counts) or not np.isfinite(counts).all():
        raise ValueError('source counts must be a finite nonempty vector')
    if (counts < 0).any() or (counts != np.floor(counts)).any() or not (counts > 0).any():
        raise ValueError('source counts must be nonnegative integers with a present class')
    if not np.isfinite(power) or not 0 <= power <= 1:
        raise ValueError('source class weight power must be in [0,1]')
    present = counts > 0
    weights = np.zeros_like(counts)
    weights[present] = counts[present] ** (-power)
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def source_weighted_mean(per_sample_loss, labels, class_weights=None):
    """Array/Tensor compatible. Source labels must have positive class weights."""
    if per_sample_loss.ndim != 1 or labels.shape != per_sample_loss.shape or not len(labels):
        raise ValueError('source loss and labels must be nonempty matching vectors')
    if class_weights is None:
        return per_sample_loss.mean()
    weights = class_weights[labels]
    return (per_sample_loss * weights).sum() / weights.sum()


def mixed_source_target_ce(per_sample_loss, labels, acceptance, source_count, class_weights=None):
    """Reweight source classes without reweighting target sample coefficients.

    L = (Ns * sum(w_y * CE_s)/sum(w_y) + sum(mask_t * CE_t))
        / (Ns + sum(mask_t)). Source acceptance must be 1, as in train.py.
    None restores the original joint masked mean exactly.
    """
    if per_sample_loss.ndim != 1 or labels.shape != per_sample_loss.shape or acceptance.shape != labels.shape:
        raise ValueError('mixed classification inputs must be matching vectors')
    if not 0 < source_count <= len(labels):
        raise ValueError('source_count must select a nonempty source prefix')
    if class_weights is None:
        return (per_sample_loss * acceptance).sum() / acceptance.sum()
    source_mean = source_weighted_mean(per_sample_loss[:source_count], labels[:source_count], class_weights)
    target_mask = acceptance[source_count:]
    target_sum = (per_sample_loss[source_count:] * target_mask).sum()
    return (source_count * source_mean + target_sum) / (source_count + target_mask.sum())
