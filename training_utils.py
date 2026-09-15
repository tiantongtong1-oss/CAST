"""Reproducible target-stage setup and bounded class balancing utilities."""

import hashlib
import math
import os
import random

import numpy as np
import torch


LOADER_STREAMS = ('source', 'prototype', 'target', 'threshold', 'validation', 'test')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    del worker_id
    # DataLoader already seeds torch. Seed other augmentation libraries from
    # that worker's epoch-specific seed, not a fixed worker-id-only seed.
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def reset_target_rng(seed, generators):
    """Give target training the same starting RNG after either source path."""
    seed_everything(seed)
    for stream_id, name in enumerate(LOADER_STREAMS):
        generators[name].manual_seed(seed + 1009 * (stream_id + 1))


def make_loader_generators(seed):
    generators = {name: torch.Generator() for name in LOADER_STREAMS}
    for stream_id, name in enumerate(LOADER_STREAMS):
        generators[name].manual_seed(seed + 1009 * (stream_id + 1))
    return generators


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def path_order_sha256(paths, root):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(os.path.relpath(path, root).replace(os.sep, '/').encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def source_class_weights(counts, power=0.0, max_ratio=2.0):
    """Relative weights from labeled source counts ONLY, bounded by max_ratio.

    The most frequent observed class has weight 1. Unobserved classes get 0;
    no synthetic prior or target pseudo-label frequency is used.
    """
    if not math.isfinite(power) or power < 0:
        raise ValueError('source balance power must be finite and nonnegative')
    if not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError('source balance max_ratio must be finite and >= 1')
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if counts.ndim != 1 or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('source counts must be a finite nonnegative vector')
    present = counts > 0
    if not present.any():
        raise ValueError('at least one labeled source class is required')
    weights = torch.zeros_like(counts)
    weights[present] = (counts.max() / counts[present]).pow(power).clamp(max=max_ratio)
    return weights


def balance_source_losses(per_sample_loss, labels, class_weights):
    """Normalize within the source batch, preserving its total sample weight.

    The target CE and the original source+accepted-target denominator are left
    intact. All-one weights recover unweighted source CE exactly.
    """
    weights = class_weights.to(per_sample_loss).index_select(0, labels.long())
    if torch.any(weights <= 0):
        raise ValueError('source batch contains a class absent from source counts')
    return per_sample_loss * (weights / weights.mean().clamp_min(1e-12))


def target_pseudo_class_weights(counts, power=0.0, max_ratio=2.0):
    """Bounded inverse-frequency weights from accepted target pseudo labels.

    This uses no target ground-truth labels. Counts may represent an epoch or a
    mini-batch; recovery v4 intentionally uses the current accepted pseudo-label
    mini-batch with a small power and a tight cap. The weighted mean under the
    observed pseudo-label distribution is normalized to one so target CE scale
    stays close to the unweighted objective. Unseen classes receive the maximum
    raw weight, ready for a newly recovered minority pseudo class.
    """
    if not math.isfinite(power) or power < 0:
        raise ValueError('target balance power must be finite and nonnegative')
    if not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError('target balance max_ratio must be finite and >= 1')

    counts = torch.as_tensor(counts, dtype=torch.float32)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError('target pseudo counts must be a nonempty vector')
    if not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('target pseudo counts must be finite and nonnegative')
    if power == 0.0:
        return torch.ones_like(counts)

    present = counts > 0
    if not present.any():
        return torch.ones_like(counts)

    max_count = counts[present].max()
    raw = torch.full_like(counts, float(max_ratio))
    raw[present] = (max_count / counts[present]).pow(power).clamp(max=max_ratio)

    normalizer = (
        (raw[present] * counts[present]).sum()
        / counts[present].sum().clamp_min(1.0)
    )
    return raw / normalizer.clamp_min(1e-12)


def balance_target_losses(per_sample_loss, labels, class_weights):
    """Apply fixed per-class pseudo-label weights to target per-sample CE."""
    if per_sample_loss.ndim != 1 or labels.ndim != 1:
        raise ValueError('target losses and labels must be one-dimensional')
    if per_sample_loss.numel() != labels.numel():
        raise ValueError('target losses and labels must have the same length')
    weights = torch.as_tensor(class_weights, dtype=per_sample_loss.dtype,
                              device=per_sample_loss.device)
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError('target class weights must be a nonempty vector')
    sample_weights = weights.index_select(0, labels.long())
    if not torch.isfinite(sample_weights).all() or torch.any(sample_weights <= 0):
        raise ValueError('target class weights must be finite and positive')
    return per_sample_loss * sample_weights
