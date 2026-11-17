import unittest
from emotion_radius import balanced_quantile, grouped_radii, POSITIVE_CLASSES, NEGATIVE_CLASSES


class GroupQuantileTests(unittest.TestCase):
    def test_groups_partition_seven_labels(self):
        self.assertEqual(POSITIVE_CLASSES, (0, 3, 6))
        self.assertEqual(NEGATIVE_CLASSES, (1, 2, 4, 5))
        self.assertEqual(sorted(POSITIVE_CLASSES + NEGATIVE_CLASSES), list(range(7)))

    def test_class_balance_and_duplication_invariance(self):
        distances = {0: [0.] * 100, 3: [1., 2.], 6: [3., 4.]}
        q = balanced_quantile(distances, POSITIVE_CLASSES, .9)
        self.assertEqual(q, 4.)
        distances[0] *= 10
        self.assertEqual(balanced_quantile(distances, POSITIVE_CLASSES, .9), q)

    def test_negative_relaxation_and_floor(self):
        d = {0: list(range(100)), 1: list(range(100))}
        positive, negative = grouped_radii(d, 92.)
        self.assertEqual(positive, 89.)
        self.assertEqual(negative, 94.)
        self.assertEqual(grouped_radii(d, 98.)[1], 98.)

    def test_empty_groups_and_invalid_distances(self):
        self.assertEqual(grouped_radii({}, 0.), (None, None))
        with self.assertRaises(ValueError):
            balanced_quantile({1: [float('nan')]}, NEGATIVE_CLASSES, .95)


try:
    import torch
    from fused_knn import FusedKNNBank
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class GroupRoutingTests(unittest.TestCase):
    def test_each_label_routes_to_group_and_own_medoid(self):
        bank = FusedKNNBank(2)
        bank.class_medoids[:] = torch.tensor([1., 0.])
        bank.class_center_ready.fill_(True)
        bank.group_radius_ready.fill_(True)
        bank.group_radii[:] = torch.tensor([.1, .5])
        queries = torch.tensor([[1., .25]]).repeat(7, 1)
        self.assertEqual(bank.radius_accepts(queries, torch.arange(7)).tolist(),
                         [False, True, True, False, True, True, False])
        bank.class_medoids[1] = torch.tensor([-1., 0.])
        self.assertFalse(bank.radius_accepts(queries[:1], torch.tensor([1])).item())

    def test_empty_refresh_clears_group_readiness(self):
        bank = FusedKNNBank(2)
        bank.group_radius_ready.fill_(True)
        bank.set_source_memory(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        self.assertFalse(bank.group_radius_ready.any().item())


if __name__ == '__main__':
    unittest.main()
