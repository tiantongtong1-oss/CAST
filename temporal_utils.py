"""Sample-indexed temporal reliability and soft pseudo-label supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPredictionBank(nn.Module):
    """Track predictions across epochs without reading target ground truth.

    The caller visits each dataset index once per epoch. Reliability is checked
    against the PREVIOUS history, before the current prediction updates it.
    Only consecutive dual-view-confident predictions extend a label's streak.
    Unreliable observations still update the probability memory, allowing a
    previously wrong class to be corrected instead of remaining locked in.
    """

    def __init__(self, num_samples, num_classes, momentum=0.7, min_streak=2):
        super().__init__()
        if num_samples < 1 or num_classes < 2:
            raise ValueError('require positive num_samples and num_classes >= 2')
        if not 0.0 <= momentum < 1.0:
            raise ValueError('temporal momentum must be in [0, 1)')
        if min_streak < 1:
            raise ValueError('temporal min_streak must be positive')
        self.momentum = float(momentum)
        self.min_streak = int(min_streak)
        self.register_buffer('probabilities', torch.zeros(num_samples, num_classes))
        self.register_buffer('seen', torch.zeros(num_samples, dtype=torch.bool))
        self.register_buffer('last_labels', torch.full((num_samples,), -1, dtype=torch.long))
        self.register_buffer('streaks', torch.zeros(num_samples, dtype=torch.long))
        self.register_buffer('last_epochs', torch.full((num_samples,), -1, dtype=torch.long))

    @torch.no_grad()
    def update_and_select(self, indices, logits1, logits2, pseudo_labels,
                          dual_mask, epoch, warmup_epochs=2):
        indices = indices.to(device=self.probabilities.device, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() != logits1.size(0):
            raise ValueError('indices must identify each prediction exactly once')
        if indices.unique().numel() != indices.numel():
            raise ValueError('duplicate dataset indices in temporal batch')
        previous_epochs = self.last_epochs[indices]
        if torch.any(previous_epochs >= epoch):
            raise ValueError('temporal samples must be visited once per increasing epoch')

        probabilities = 0.5 * (F.softmax(logits1.detach(), dim=1)
                               + F.softmax(logits2.detach(), dim=1))
        previous = self.probabilities[indices]
        seen = self.seen[indices]
        labels = pseudo_labels.detach().long()
        dual_mask = dual_mask.detach().bool()
        same_label = self.last_labels[indices].eq(labels)
        consecutive = previous_epochs.eq(epoch - 1)
        streaks = torch.where(
            dual_mask,
            torch.where(same_label & consecutive, self.streaks[indices] + 1,
                        torch.ones_like(indices)),
            torch.zeros_like(indices),
        )
        history_agrees = seen & previous.argmax(dim=1).eq(labels)
        stable_mask = dual_mask & history_agrees & (streaks >= self.min_streak)
        selected = dual_mask if epoch < warmup_epochs else stable_mask

        updated = self.momentum * previous + (1.0 - self.momentum) * probabilities
        updated = torch.where(seen.unsqueeze(1), updated, probabilities)
        updated = updated / updated.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self.probabilities[indices] = updated
        self.seen[indices] = True
        self.last_labels[indices] = labels
        self.streaks[indices] = streaks
        self.last_epochs[indices] = int(epoch)

        stats = {
            'dual_accepted': int(dual_mask.sum().item()),
            'stable_accepted': int(stable_mask.sum().item()),
            'history_disagrees': int((dual_mask & seen & ~history_agrees).sum().item()),
            'label_flips': int((seen & ~same_label).sum().item()),
        }
        return selected.float(), updated.detach(), stats


def mixed_target_cross_entropy(logits, labels, soft_targets, soft_weight=0.5):
    """Per-sample hard/soft CE; the caller applies the final reliability mask.

    Manual soft CE supports the repository's older PyTorch versions. Teacher
    distributions are detached, never sharpened, and affect target CE only.
    """
    if not 0.0 <= soft_weight <= 1.0:
        raise ValueError('soft_weight must be in [0, 1]')
    hard_loss = F.cross_entropy(logits, labels, reduction='none')
    if soft_weight == 0.0:
        return hard_loss
    soft_loss = -(soft_targets.detach() * F.log_softmax(logits, dim=1)).sum(dim=1)
    return (1.0 - soft_weight) * hard_loss + soft_weight * soft_loss
