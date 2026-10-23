import unittest
import numpy as np
from source_class_weights import domain_normalized_classification_loss


class ArrayTensor(np.ndarray):
    """NumPy arithmetic adapter for CPU formula tests without PyTorch installed."""
    def numel(self):
        return self.size

    def clamp_min(self, minimum):
        return np.maximum(self, minimum).view(ArrayTensor)


def tensor(values):
    return np.asarray(values, dtype=np.float64).view(ArrayTensor)


class DomainNormalizedLossTest(unittest.TestCase):
    def loss(self, losses, weights, mask):
        return float(domain_normalized_classification_loss(
            tensor(losses), tensor(weights), tensor(mask)))

    def test_unit_weights_recover_v22_with_rejected_targets(self):
        self.assertAlmostEqual(self.loss([2, 4, 6, 100], [1, 1], [1, 0]), 4)

    def test_weighted_source_preserves_domain_counts(self):
        # Source mean=(2*1+4*3)/4=3.5; accepted target mean=6.
        self.assertAlmostEqual(self.loss([2, 4, 6, 100], [1, 3], [1, 0]), 13 / 3)

    def test_no_accepted_target_and_empty_target(self):
        self.assertAlmostEqual(self.loss([2, 4, 99], [1, 3], [0]), 3.5)
        self.assertAlmostEqual(self.loss([2, 4], [1, 3], []), 3.5)

    def test_source_weight_scale_does_not_change_domain_mix(self):
        self.assertAlmostEqual(self.loss([2, 4, 6], [1, 3], [1]),
                               self.loss([2, 4, 6], [0.1, 0.3], [1]))

    def test_torch_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable; NumPy formula cases still run')
        losses = torch.tensor([2., 4., 6., 100.], requires_grad=True)
        result = domain_normalized_classification_loss(
            losses, torch.tensor([1., 3.]), torch.tensor([1., 0.]))
        result.backward()
        torch.testing.assert_allclose(result, torch.tensor(13 / 3))
        torch.testing.assert_allclose(losses.grad, torch.tensor([1/6, 1/2, 1/3, 0.]))


if __name__ == '__main__':
    unittest.main()
