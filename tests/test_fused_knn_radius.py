import unittest
from pathlib import Path
try:
    import torch
    from fused_knn import FusedKNNBank
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class SharedRadiusTests(unittest.TestCase):
    def test_balanced_quantile_and_label_specific_centers(self):
        bank = FusedKNNBank(2, 3, k=2)
        features = torch.tensor([[1., 0.]] * 30 + [[0., 1.], [1., 1.], [1., 0.]])
        bank.set_source_memory(features, torch.tensor([0] * 30 + [1] * 3))
        z = torch.nn.functional.normalize(features[-3:], dim=1)
        expected = (z.double() - z[1].double()).norm(dim=1).max()
        torch.testing.assert_close(bank.global_radius, expected.float())
        self.assertEqual(bank.class_counts.tolist(), [30, 3, 0])
        self.assertEqual(bank.class_center_ready.tolist(), [True, True, False])
        bank.global_radius.fill_(.1)
        query = torch.tensor([[5., 0.]] * 5)
        self.assertEqual(bank.radius_accepts(query, torch.tensor([0, 1, 2, -1, 3])).tolist(),
                         [True, False, False, False, False])

    def test_scale_invariance_and_class_duplication(self):
        features = torch.tensor([[1., 0.], [2., .4], [3., .9],
                                 [0., 1.], [.2, 2.], [.6, 3.]])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        bank = FusedKNNBank(2, 2)
        bank.set_source_memory(features, labels)
        centers, radius = bank.class_medoids.clone(), bank.global_radius.clone()
        bank.set_source_memory(features * torch.tensor([1., 2., 3., 4., 5., 6.])[:, None], labels)
        torch.testing.assert_close(bank.class_medoids, centers)
        torch.testing.assert_close(bank.global_radius, radius)
        bank.set_source_memory(torch.cat((features[:3].repeat(3, 1), features[3:])),
                               torch.tensor([0] * 9 + [1] * 3))
        torch.testing.assert_close(bank.global_radius, radius)

    def test_refresh_singleton_empty_and_strict_zero_radius(self):
        bank = FusedKNNBank(2, 2)
        bank.set_source_memory(torch.tensor([[1., 0.], [1., .2], [0., 1.], [.2, 1.]]),
                               torch.tensor([0, 0, 1, 1]))
        bank.set_source_memory(torch.tensor([[1., 0.], [2., 0.], [0., 1.]]),
                               torch.tensor([0, 0, 1]))
        self.assertEqual(bank.class_center_ready.tolist(), [True, False])
        self.assertEqual(bank.global_radius.item(), 0.)
        self.assertFalse(bank.radius_accepts(torch.tensor([[1., 0.]]), torch.tensor([0])).item())
        bank.set_source_memory(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        self.assertFalse(bank.radius_ready.item())


class SharedIntegrationTests(unittest.TestCase):
    def test_training_wiring_and_typo_fixes(self):
        source = (Path(__file__).resolve().parents[1] / 'train.py').read_text()
        self.assertIn('radius_pass_mask = reliability_bank.radius_accepts(', source)
        self.assertIn('scores = torch.cat((conf1, conf2), dim=0)', source)
        self.assertNotIn('doubsle', source)
        self.assertIn('fusion_candidates = confidence_candidates | radius_candidates', source)


if __name__ == '__main__':
    unittest.main()
