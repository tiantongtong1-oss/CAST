import unittest
import torch
from raw_medoid_quantile import RawMedoidQuantileBank, scaled_class_thresholds
from ema_utils import select_dual_view_pseudo_labels


class RawQuantileTests(unittest.TestCase):
    def test_lambda_times_class_mean_and_missing_class(self):
        thresholds, means = scaled_class_thresholds(
            torch.tensor([1.6, .6, 0.]), torch.tensor([2., 1., 0.]), 1.1)
        torch.testing.assert_close(thresholds[:2], torch.tensor([.88, .66]))
        self.assertTrue(torch.isinf(thresholds[2]))
        high, _ = scaled_class_thresholds(torch.tensor([.9]), torch.tensor([1.]), 2.)
        self.assertGreater(high.item(), 1.)

    def bank(self):
        bank = RawMedoidQuantileBank(torch.eye(2), torch.ones(2, dtype=torch.bool))
        features = torch.tensor([[1., 0.], [2., 0.], [3., 0.], [4., 0.], [8., 0.]])
        bank.set_source_memory(features, torch.tensor([0, 0, 1, 1, 1]))
        return bank, features

    def test_raw_medoid_and_q90(self):
        bank, features = self.bank()
        torch.testing.assert_close(bank.memory_features, features)
        torch.testing.assert_close(bank.medoid, features[2])
        distances = torch.tensor([2., 1., 0., 1., 5.], dtype=torch.float64)
        torch.testing.assert_close(bank.global_radius, torch.quantile(distances, .9).float())
        self.assertTrue(bank.accepts(torch.tensor([[3., 0.]]), torch.tensor([0])).item())
        self.assertFalse(bank.accepts(torch.tensor([[8., 0.]]), torch.tensor([0])).item())
        # Same direction, different length: angular normalization would collapse these.
        bank.global_radius.zero_()
        self.assertFalse(bank.accepts(features[2:3], torch.tensor([0])).item())

    def test_knn_uses_raw_euclidean_not_cosine(self):
        bank, _ = self.bank()
        passed, checked = bank.neighbor_accepts(torch.tensor([[8., 0.]]), torch.tensor([1]),
                                                torch.tensor([True]), k=3)
        self.assertTrue(passed.item())
        self.assertTrue(checked.item())

    def test_both_original_and_augmented_must_pass(self):
        confident = torch.tensor([[3., 0.]])
        uncertain = torch.tensor([[.1, 0.]])
        _, mask, _ = select_dual_view_pseudo_labels(confident, uncertain, torch.tensor([.8, .8]))
        self.assertFalse(mask.bool().item())


if __name__ == '__main__':
    unittest.main()
