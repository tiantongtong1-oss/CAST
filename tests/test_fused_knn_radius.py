import unittest
try:
    import torch
    from fused_knn import FusedKNNBank
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class RadiusFusionTests(unittest.TestCase):
    def test_normalized_q90_radius_and_query(self):
        bank = FusedKNNBank(2, 2, k=2)
        features = torch.tensor([[1., 0.], [2., .2], [3., .6], [0., 4.]])
        bank.set_source_memory(features, torch.tensor([0, 0, 0, 1]))
        distances = (bank.source_features.double() - bank.medoid.double()).norm(dim=1)
        torch.testing.assert_close(bank.global_radius, torch.quantile(distances, .9).float())
        query = bank.medoid[None] * 9
        self.assertTrue(bank.radius_accepts(query).item())
        self.assertFalse(bank.radius_accepts(torch.zeros(1, 2)).item())
        bank.global_radius.zero_()
        self.assertFalse(bank.radius_accepts(bank.medoid[None]).item())


if __name__ == '__main__':
    unittest.main()
