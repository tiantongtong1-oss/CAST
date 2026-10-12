import math
import unittest
import torch
import torch.nn.functional as F
from prototype_radius_reliability import PrototypeRadiusKNNReliabilityBank
from sparse_reliable_knn import SparseReliableKNNBank


def points(angles):
    return torch.tensor([[math.cos(a), math.sin(a)] for a in angles])


class RadiusTests(unittest.TestCase):
    def test_exact_farthest_k_mean_and_batch_invariance(self):
        x = points([0., .1, .2, .6, 1.2, 2.])
        labels = torch.tensor([0, 0, 0, 0, 1, 1])
        banks = []
        for batch_size in (1, 2, 6):
            bank = PrototypeRadiusKNNReliabilityBank(3, 2, source_radius_k=2)
            bank.begin_source_refresh()
            for start in range(0, len(x), batch_size):
                bank.accumulate_source(x[start:start+batch_size], labels[start:start+batch_size])
            bank.finalize_source()
            for c in (0, 1):
                z = F.normalize(x[labels == c], dim=1)
                mu = z.double().mean(0).float()
                radius = (z - mu).norm(dim=1).topk(2).values.mean()
                torch.testing.assert_close(bank.class_means[c], mu)
                torch.testing.assert_close(bank.class_radii[c], radius)
            self.assertFalse(bank.distribution_initialized[2])
            self.assertEqual(len(bank._source_batches), 0)
            banks.append(bank)
        for bank in banks[1:]:
            torch.testing.assert_close(bank.class_radii, banks[0].class_radii)

    def test_k_larger_than_class_and_degenerate_class(self):
        bank = PrototypeRadiusKNNReliabilityBank(2, 2, source_radius_k=20)
        x = points([0., .2, .9])
        bank.accumulate_source(x, torch.zeros(3, dtype=torch.long))
        bank.accumulate_source(points([1.]), torch.tensor([1]))
        bank.finalize_source()
        torch.testing.assert_close(bank.class_radii[0], (x - x.mean(0)).norm(dim=1).mean())
        self.assertEqual(bank.class_radii[1].item(), 0.)
        self.assertFalse(bank.in_distribution(points([1.]), torch.tensor([1])).item())

    def test_assigned_class_strict_boundary_and_invalid(self):
        bank = PrototypeRadiusKNNReliabilityBank(2, 2)
        bank.class_means.copy_(points([0., math.pi]))
        bank.class_radii.fill_(.5)
        bank.distribution_initialized.fill_(True)
        x = points([.1])
        self.assertTrue(bank.in_distribution(x, torch.tensor([0])).item())
        self.assertFalse(bank.in_distribution(x, torch.tensor([1])).item())
        bank.class_radii[0] = bank.distribution_distance(x, torch.tensor([0]))[0]
        self.assertFalse(bank.in_distribution(x, torch.tensor([0])).item())
        bad = torch.tensor([[0., 0.], [float('nan'), 1.], [1., 0.], [1., 0.]])
        self.assertFalse(bank.in_distribution(bad, torch.tensor([0, 0, -1, 2])).any())
        self.assertFalse(bank.in_distribution(x, torch.tensor([0])).requires_grad)

    def test_refresh_replaces_threshold_and_missing_class(self):
        bank = PrototypeRadiusKNNReliabilityBank(2, 2, source_radius_k=1)
        bank.accumulate_source(points([0., .1, 1., 1.5]), torch.tensor([0, 0, 1, 1]))
        bank.finalize_source()
        first = bank.class_radii[0].clone()
        bank.begin_source_refresh()
        bank.accumulate_source(points([0., .8]), torch.tensor([0, 0]))
        bank.finalize_source()
        self.assertGreater(bank.class_radii[0].item(), first.item())
        self.assertFalse(bank.distribution_initialized[1])
        self.assertFalse(bank.in_distribution(points([1.]), torch.tensor([1])).item())
        self.assertEqual(bank.source_count.item(), 2)

    def test_checkpoint_contains_radius_not_variance(self):
        bank = SparseReliableKNNBank(2, 2)
        state = bank.state_dict()
        self.assertIn('class_radii', state)
        self.assertFalse(any('var' in key or 'sigma' in key or 'mahalanobis' in key for key in state))
        other = SparseReliableKNNBank(2, 2)
        other.load_state_dict(state)
        torch.testing.assert_close(other.class_radii, bank.class_radii)

    def test_bad_radius_k(self):
        for k in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                PrototypeRadiusKNNReliabilityBank(2, 2, source_radius_k=k)


if __name__ == '__main__':
    unittest.main()
