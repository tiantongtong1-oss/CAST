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

    def test_normalized_medoid_and_q90_preserve_raw_memory(self):
        bank, _ = self.bank()
        angles = torch.tensor([-60., -20., 0., 30., 100.], dtype=torch.float64)
        angles = angles * (torch.pi / 180)
        unit = torch.stack((angles.cos(), angles.sin()), dim=1)
        features = unit * torch.tensor([1., 2., 3., 4., 8.])[:, None]
        bank.set_source_memory(features, torch.tensor([0, 0, 1, 1, 1]))
        torch.testing.assert_close(bank.memory_features, features.float())
        self.assertEqual(bank.medoid_memory_index.item(), 2)
        torch.testing.assert_close(bank.medoid, torch.tensor([1., 0.]))
        normalized = torch.nn.functional.normalize(features.float(), dim=1).double()
        distances = (normalized - normalized[2]).norm(dim=1)
        torch.testing.assert_close(bank.global_radius, torch.quantile(distances, .9).float())
        # Changing source magnitudes must not alter the radius or center.
        other, _ = self.bank()
        other.set_source_memory(features * torch.tensor([8., 3., 2., 5., 1.])[:, None],
                                torch.tensor([0, 0, 1, 1, 1]))
        torch.testing.assert_close(other.medoid, bank.medoid)
        torch.testing.assert_close(other.global_radius, bank.global_radius)

    def test_target_normalization_full_radius_and_strict_boundary(self):
        bank, _ = self.bank()
        bank.medoid.copy_(torch.tensor([1., 0.]))
        bank.global_radius.fill_(1.)
        # Unit-vector distance is 0.8: passes R but would fail 0.7R.
        query = torch.tensor([[.68, (1. - .68 ** 2) ** .5]])
        self.assertTrue(bank.accepts(query, torch.tensor([0])).item())
        self.assertTrue(bank.accepts(query * 10, torch.tensor([0])).item())
        self.assertFalse(bank.accepts(torch.tensor([[-2., 0.]]), torch.tensor([0])).item())
        bank.global_radius.zero_()
        self.assertFalse(bank.accepts(torch.tensor([[3., 0.]]), torch.tensor([0])).item())

    def test_invalid_features_and_insufficient_memory(self):
        bank, _ = self.bank()
        queries = torch.tensor([[0., 0.], [float('nan'), 1.], [float('inf'), 1.]])
        self.assertFalse(bank.accepts(queries, torch.zeros(3, dtype=torch.long)).any().item())
        bank.set_source_memory(torch.tensor([[1., 0.]]), torch.tensor([0]))
        self.assertFalse(bank.global_ready.item())
        self.assertFalse(bank.accepts(torch.tensor([[1., 0.]]), torch.tensor([0])).item())

    def test_knn_normalizes_source_and_query(self):
        bank, _ = self.bank()
        features = torch.tensor([[1., .1], [2., .4], [3., .9],
                                 [0., 1.], [.1, 2.], [.2, 3.]])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        bank.set_source_memory(features, labels)
        query = torch.tensor([[100., 0.]])
        passed, checked = bank.neighbor_accepts(query, torch.tensor([0]),
                                                torch.tensor([True]), k=3)
        self.assertTrue(passed.item())
        self.assertTrue(checked.item())
        bank.set_source_memory(features * torch.tensor([50., 2., 8., 1., 7., 3.])[:, None], labels)
        scaled, _ = bank.neighbor_accepts(query * .01, torch.tensor([0]),
                                         torch.tensor([True]), k=3)
        torch.testing.assert_close(scaled, passed)

    def test_both_original_and_augmented_must_pass(self):
        confident = torch.tensor([[3., 0.]])
        uncertain = torch.tensor([[.1, 0.]])
        _, mask, _ = select_dual_view_pseudo_labels(confident, uncertain, torch.tensor([.8, .8]))
        self.assertFalse(mask.bool().item())


if __name__ == '__main__':
    unittest.main()
