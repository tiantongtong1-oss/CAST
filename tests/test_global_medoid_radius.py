import unittest
import torch
from global_medoid_radius import GlobalMedoidRadiusBank, exact_medoid_index


class MedoidTests(unittest.TestCase):
    def test_exact_euclidean_and_chunking(self):
        # Euclidean medoid is 1, whereas nearest-to-mean is 2: distinguishes
        # the requested objective from squared-distance/mean substitutes.
        z = torch.tensor([[0.], [1.], [2.], [100.]], dtype=torch.float64)
        reference = (z[:, None] - z[None, :]).norm(dim=-1).sum(dim=1).argmin()
        for chunk in (1, 2, 8):
            self.assertEqual(exact_medoid_index(z, chunk).item(), reference.item())

    def test_global_radius_matches_brute_force_and_is_label_independent(self):
        angles = torch.tensor([0., .1, .2, .4, .7, 1., 1.5, 2., 2.8])
        features = torch.stack([angles.cos(), angles.sin()], dim=1)
        bank = GlobalMedoidRadiusBank(torch.eye(2), torch.ones(2, dtype=torch.bool))
        bank.set_source_memory(features, torch.arange(9) % 2)
        z = bank.memory_features.double()
        idx = torch.cdist(z, z).sum(1).argmin()
        torch.testing.assert_close(bank.medoid, bank.memory_features[idx])
        expected = (z - z[idx]).norm(dim=1).topk(7).values.mean().float()
        torch.testing.assert_close(bank.global_radius, expected)
        query = features[:1].repeat(2, 1)
        result = bank.accepts(query, torch.tensor([0, 1]))
        self.assertEqual(result[0].item(), result[1].item())
        bank.global_radius.zero_()
        self.assertFalse(bank.accepts(bank.medoid[None], torch.tensor([0])).item())

    def test_insufficient_data_and_invalid_vectors(self):
        bank = GlobalMedoidRadiusBank(torch.eye(2), torch.ones(2, dtype=torch.bool))
        bank.set_source_memory(torch.tensor([[1., 0.]]).repeat(6, 1), torch.zeros(6, dtype=torch.long))
        self.assertFalse(bank.global_ready.item())
        bank.set_source_memory(torch.tensor([[1., 0.]]).repeat(7, 1), torch.zeros(7, dtype=torch.long))
        bank.global_radius.fill_(1)
        self.assertEqual(bank.accepts(torch.tensor([[0., 0.], [float('nan'), 1.], [1., 0.]]),
                                     torch.tensor([0, 0, -1])).tolist(), [False]*3)


if __name__ == '__main__':
    unittest.main()
