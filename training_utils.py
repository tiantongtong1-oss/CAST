"""Training math shared by the runner and CPU regression tests."""

import random

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def update_ema(student, teacher, decay=0.999):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.mul_(decay).add_(student_param, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if teacher_buffer.is_floating_point():
            teacher_buffer.mul_(decay).add_(student_buffer, alpha=1.0 - decay)
        else:
            teacher_buffer.copy_(student_buffer)


def target_affinity_weight(epoch, maximum, warmup=5, ramp=5):
    """A real zero disables affinity in every epoch, including warmup/ramp."""
    return maximum * min(1.0, max(0.0, (epoch - warmup + 1) / max(ramp, 1)))


@torch.no_grad()
def select_pseudo_labels(logits1, logits2, epoch, epochs, phi,
                         threshold_min=0.8, threshold_max=0.95):
    """CATM on mean probabilities; BOTH views must clear their class threshold.

    Target ground-truth labels deliberately are not inputs to this function.
    Missing classes get the conservative ceiling, not a zero threshold.
    """
    prob1, prob2 = logits1.softmax(1), logits2.softmax(1)
    mean_prob = (prob1 + prob2) / 2
    confidence, labels = mean_prob.max(1)
    n_classes = mean_prob.shape[1]
    counts = torch.bincount(labels, minlength=n_classes)
    sums = torch.zeros(n_classes, device=logits1.device, dtype=mean_prob.dtype)
    sums.scatter_add_(0, labels, confidence)
    thresholds = (sums / counts.clamp_min(1) * phi * epochs / max(epochs - epoch, 1))
    thresholds = thresholds.clamp(threshold_min, threshold_max)
    thresholds[counts == 0] = threshold_max
    agreement = prob1.argmax(1).eq(prob2.argmax(1))
    # Agreement alone does not mean that the second prediction is reliable.
    confidence = torch.minimum(prob1.gather(1, labels[:, None]).squeeze(1),
                               prob2.gather(1, labels[:, None]).squeeze(1))
    sample_thresholds = thresholds[labels]
    selected = agreement & (confidence > sample_thresholds)
    weights = ((confidence - sample_thresholds) / (1 - sample_thresholds).clamp_min(1e-6))
    weights = weights.clamp(0, 1) * selected
    return labels, selected, weights, thresholds, agreement


def classification_losses(logits, source_targets, pseudo_targets, pseudo_weights):
    """Separate reductions keep the target gradient from shrinking with coverage."""
    n_source = source_targets.numel()
    source_loss = F.cross_entropy(logits[:n_source], source_targets)
    target_losses = F.cross_entropy(logits[n_source:], pseudo_targets, reduction='none')
    weights = pseudo_weights.detach().to(target_losses)
    # No selected targets => exact differentiable zero, no NaNs.
    target_loss = (target_losses * weights).sum() / weights.sum().clamp_min(1e-6)
    return source_loss, target_loss


def classifier_weight_loss(weight):
    normalized = F.normalize(weight, dim=1)
    identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
    return ((normalized @ normalized.t() - identity + 1) / 2).mean()
