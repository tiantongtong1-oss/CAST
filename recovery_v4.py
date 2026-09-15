"""Recovery-v4 target-stage runner.

The base trainer deliberately ignores FER train ground truth.  This wrapper
keeps that invariant and adds a conservative, per-mini-batch reweighting of
accepted target pseudo-label CE so dominant pseudo classes do not monopolize
the target gradient.  It is opt-in: existing train.py and recovery-v3 behavior
remain unchanged.
"""

import os

import torch

import train
from temporal_utils import mixed_target_cross_entropy as base_mixed_target_cross_entropy
from training_utils import balance_target_losses, target_pseudo_class_weights


TARGET_BALANCE_POWER = float(os.environ.get('CAST_TARGET_BALANCE_POWER', '0.20'))
TARGET_BALANCE_MAX_RATIO = float(os.environ.get('CAST_TARGET_BALANCE_MAX_RATIO', '1.50'))


def balanced_mixed_target_cross_entropy(logits, labels, soft_targets, soft_weight=0.5):
    """Mixed target CE with mild pseudo-class balancing.

    Weights are derived only from the accepted pseudo labels in the current
    mini-batch.  No FER train/test labels are read.  power=0 exactly recovers
    the original target objective; v4 uses a small 0.20 power and 1.5x cap to
    avoid amplifying noisy minority pseudo labels too aggressively.
    """
    losses = base_mixed_target_cross_entropy(
        logits, labels, soft_targets, soft_weight=soft_weight
    )
    if TARGET_BALANCE_POWER <= 0.0 or labels.numel() == 0:
        return losses

    counts = torch.bincount(labels.detach().long(), minlength=logits.size(1))
    weights = target_pseudo_class_weights(
        counts,
        power=TARGET_BALANCE_POWER,
        max_ratio=TARGET_BALANCE_MAX_RATIO,
    )
    return balance_target_losses(losses, labels, weights)


def main():
    print('Recovery v4 target pseudo balance: power=%.3f max_ratio=%.3f' %
          (TARGET_BALANCE_POWER, TARGET_BALANCE_MAX_RATIO))
    train.mixed_target_cross_entropy = balanced_mixed_target_cross_entropy
    return train.run_training()


if __name__ == '__main__':
    main()
