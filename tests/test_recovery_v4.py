import unittest

import torch
import torch.nn.functional as F

from recovery_v4 import balanced_mixed_target_cross_entropy
from training_utils import balance_target_losses, target_pseudo_class_weights


class TargetPseudoBalanceTests(unittest.TestCase):
    def test_weights_are_bounded_and_preserve_observed_mean_scale(self):
        counts = torch.tensor([10., 2., 0., 20.])
        weights = target_pseudo_class_weights(counts, power=0.5, max_ratio=1.5)
        present = counts > 0
        weighted_mean = (weights[present] * counts[present]).sum() / counts[present].sum()
        self.assertAlmostEqual(weighted_mean.item(), 1.0, places=6)
        self.assertLessEqual((weights.max() / weights.min()).item(), 1.5 + 1e-6)
        self.assertGreater(weights[1].item(), weights[3].item())

    def test_power_zero_recovers_original_target_loss(self):
        import recovery_v4

        logits = torch.tensor([[2., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        labels = torch.tensor([0, 0, 1])
        soft = F.softmax(logits, dim=1)
        old_power = recovery_v4.TARGET_BALANCE_POWER
        try:
            recovery_v4.TARGET_BALANCE_POWER = 0.0
            actual = balanced_mixed_target_cross_entropy(logits, labels, soft, 0.1)
            hard = F.cross_entropy(logits, labels, reduction='none')
            soft_loss = -(soft * F.log_softmax(logits, dim=1)).sum(dim=1)
            expected = 0.9 * hard + 0.1 * soft_loss
            self.assertTrue(torch.allclose(actual, expected))
        finally:
            recovery_v4.TARGET_BALANCE_POWER = old_power

    def test_minority_pseudo_class_gets_larger_gradient_weight(self):
        losses = torch.ones(4)
        labels = torch.tensor([0, 0, 0, 1])
        weights = target_pseudo_class_weights([3, 1], power=0.2, max_ratio=1.5)
        balanced = balance_target_losses(losses, labels, weights)
        self.assertGreater(balanced[3].item(), balanced[0].item())
        self.assertLessEqual((balanced[3] / balanced[0]).item(), 1.5 + 1e-6)


if __name__ == '__main__':
    unittest.main()
