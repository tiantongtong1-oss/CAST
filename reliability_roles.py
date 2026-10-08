"""Sample roles and optional soft supervision; no target labels are consumed."""

import torch
from torch import nn
import torch.nn.functional as F


class ReliabilityRoles(nn.Module):
    """Promote only after consecutive eligible epochs with an unchanged label.

    Stable dataset IDs index persistent history. Repeated observations within one
    epoch never increment the streak; an ineligible observation revokes the role.
    Neighbor roles are snapshotted at refresh, so rescue cannot feed itself within
    the same refresh. Checkpoints preserve history, but train.py remains weights-
    initialization-only (not an exact resume implementation).
    """

    def __init__(self, num_samples, promotion_epochs=3):
        super().__init__()
        if num_samples < 1 or promotion_epochs < 1:
            raise ValueError('sample count and promotion_epochs must be positive')
        self.promotion_epochs = int(promotion_epochs)
        self.register_buffer('labels', torch.full((num_samples,), -1, dtype=torch.long))
        self.register_buffer('streak', torch.zeros(num_samples, dtype=torch.long))
        self.register_buffer('last_epoch', torch.full((num_samples,), -2, dtype=torch.long))
        self.register_buffer('quality', torch.zeros(num_samples))

    def _ids(self, ids):
        ids = ids.detach().to(device=self.labels.device, dtype=torch.long)
        if ids.ndim != 1 or (ids < 0).any() or (ids >= self.labels.numel()).any():
            raise ValueError('sample IDs must index the target training dataset')
        if ids.unique().numel() != ids.numel():
            raise ValueError('sample IDs must be unique within an observation')
        return ids

    @torch.no_grad()
    def observe(self, ids, labels, eligible, confidence, epoch):
        ids = self._ids(ids)
        labels = labels.detach().to(self.labels)
        eligible = eligible.detach().to(device=ids.device, dtype=torch.bool)
        confidence = confidence.detach().to(self.quality)
        if any(x.shape != ids.shape for x in (labels, eligible, confidence)):
            raise ValueError('role observations must have matching shapes')
        if epoch < 0 or (self.last_epoch[ids] > epoch).any():
            raise ValueError('role epochs must be nonnegative and monotone')
        eligible &= torch.isfinite(confidence) & (confidence >= 0) & (confidence <= 1)
        previous_epoch = self.last_epoch[ids]
        previous_streak = self.streak[ids]
        same_label = self.labels[ids].eq(labels)
        consecutive = previous_epoch.eq(epoch - 1) & same_label & (previous_streak > 0)
        streak = torch.where(consecutive, previous_streak + 1, torch.ones_like(ids))
        # Never promote by seeing the same image repeatedly in a single epoch.
        repeated = previous_epoch.eq(epoch)
        streak = torch.where(repeated & same_label, previous_streak, streak)
        streak = torch.where(repeated & ~same_label, torch.zeros_like(streak), streak)
        streak = torch.where(eligible, streak, torch.zeros_like(streak))
        self.labels[ids] = labels
        self.streak[ids] = streak
        self.last_epoch[ids] = epoch
        self.quality[ids] = torch.where(eligible, confidence, torch.zeros_like(confidence))
        return eligible & (streak >= self.promotion_epochs)

    @torch.no_grad()
    def sender_scores(self, ids, labels, confidence, epoch):
        ids = self._ids(ids)
        labels = labels.detach().to(self.labels)
        confidence = confidence.detach().to(self.quality)
        if labels.shape != ids.shape or confidence.shape != ids.shape:
            raise ValueError('sender observations must have matching shapes')
        ready = self.streak[ids] >= self.promotion_epochs
        ready &= self.labels[ids].eq(labels) & self.last_epoch[ids].eq(epoch - 1)
        score = torch.minimum(confidence, self.quality[ids])
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
        return torch.where(ready, score, torch.zeros_like(score))


@torch.no_grad()
def neighbor_soft_targets(teacher_probs, result, agreement, hard_learn_mask, mix=0.5):
    """Only unaccepted, agreeing queries with reliable conflicting neighbors.

    Soft repair recipients never become hard-label anchors through this path.
    Stable teacher evidence in later epochs is required for normal promotion.
    """
    if not 0 < mix <= 1:
        raise ValueError('soft target mixing coefficient must be in (0, 1]')
    mask = torch.zeros_like(agreement, dtype=torch.bool)
    targets = teacher_probs.detach().clone()
    if result is None or 'neighbor_probs' not in result:
        return mask, targets
    neighbor_probs = result['neighbor_probs'].detach()
    teacher_label = teacher_probs.argmax(dim=1)
    neighbor_label = neighbor_probs.argmax(dim=1)
    mask = agreement.bool() & ~hard_learn_mask.bool()
    mask &= result['neighbor_reliable'] & neighbor_label.ne(teacher_label)
    targets[mask] = ((1 - mix) * teacher_probs[mask] + mix * neighbor_probs[mask])
    return mask, targets.detach()


def soft_supervision_loss(logits, soft_targets, mask):
    if not mask.any():
        return logits.sum() * 0.0
    return -(soft_targets[mask].detach() * F.log_softmax(logits[mask], dim=1)).sum(1).mean()
