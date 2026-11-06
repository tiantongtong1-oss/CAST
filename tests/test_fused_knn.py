import unittest
from pathlib import Path
try:
    import torch
    from fused_knn import FusedKNNBank
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class FusionTests(unittest.TestCase):
    def make_bank(self, target_probs, source_labels=(0, 0), threshold=.7):
        bank = FusedKNNBank(2, 2, threshold, k=2)
        bank.set_source_memory(torch.tensor([[2., .2], [3., .6]]),
                               torch.tensor(source_labels))
        bank.set_target_memory(torch.tensor([[1., 0.], [2., .2], [3., .6]]),
                               torch.tensor(target_probs), torch.tensor([10, 11, 12]))
        return bank

    def query(self, bank, confidence=True, agreement=True):
        return bank.select(torch.tensor([[10., 0.]]), torch.tensor([0]),
                           torch.tensor([confidence]), torch.tensor([agreement]),
                           torch.tensor([10]))

    def test_equal_fusion_soft_probabilities_and_self_exclusion(self):
        bank = self.make_bank([[0., 1.], [.8, .2], [.6, .4]])
        passed, checked, support, ps, pt, _ = self.query(bank)
        self.assertTrue(passed.item())
        self.assertTrue(checked.item())
        self.assertAlmostEqual(ps.item(), 1., places=5)
        self.assertAlmostEqual(pt.item(), .7, places=5)
        self.assertAlmostEqual(support.item(), .85, places=5)

    def test_conflicting_domains_rejected(self):
        bank = self.make_bank([[0., 1.], [0., 1.], [0., 1.]])
        passed, checked, support, _, _, disagreement = self.query(bank)
        self.assertTrue(checked.item())
        self.assertFalse(passed.item())
        self.assertTrue(disagreement.item())
        self.assertAlmostEqual(support.item(), .5)

    def test_full_neighbors_required_and_candidate_gate(self):
        bank = self.make_bank([[1., 0.]] * 3)
        for conf, agree in [(False, True), (True, False)]:
            passed, checked, *_ = self.query(bank, conf, agree)
            self.assertFalse(passed.item())
            self.assertFalse(checked.item())
        bank.set_target_memory(torch.tensor([[1., 0.], [2., .2]]),
                               torch.tensor([[1., 0.], [1., 0.]]), torch.tensor([10, 11]))
        passed, checked, *_ = self.query(bank)
        self.assertFalse(passed.item())
        self.assertFalse(checked.item())

    def test_normalization_and_argmax_gate(self):
        bank = self.make_bank([[0., 1.]] * 3, source_labels=(1, 1), threshold=.0)
        torch.testing.assert_close(bank.source_features.norm(dim=1), torch.ones(2))
        torch.testing.assert_close(bank.memory_features.norm(dim=1), torch.ones(3))
        passed, _, *_ = self.query(bank)
        self.assertFalse(passed.item())


class IntegrationTests(unittest.TestCase):
    def test_probability_memory_and_no_radius(self):
        train = (Path(__file__).resolve().parents[1] / 'train.py').read_text()
        self.assertIn('memory_predictions.append(F.softmax(logits1, dim=1).cpu())', train)
        self.assertIn('knn_support_threshold', train)
        self.assertIn('agreement_mask, sample_ids,', train)
        self.assertIn('reliability_bank.set_source_memory', train)
        self.assertNotIn('RawMedoidQuantileBank', train)


if __name__ == '__main__':
    unittest.main()
