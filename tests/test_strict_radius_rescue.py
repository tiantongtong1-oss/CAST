"""Behavioral regression tests for separate high-confidence and radius paths."""
import unittest
import torch
from fused_knn import FusedKNNBank


class StrictRescueTests(unittest.TestCase):
    def bank(self, probs):
        bank = FusedKNNBank(2, 2, support_threshold=.7, k=1)
        # Set normalized memories directly: these tests exercise selection only.
        bank.source_features = torch.tensor([[1., 0.]])
        bank.source_labels = torch.tensor([0])
        bank.memory_features = torch.tensor([[1., 0.], [1., 0.]])
        bank.memory_probs = torch.tensor([[0., 1.], probs])
        bank.memory_ids = torch.tensor([10, 11])
        return bank

    def select(self, bank, high=False, radius=True, agree=True):
        return bank.select(torch.tensor([[1., 0.]]), torch.tensor([0]),
                           torch.tensor([high]), torch.tensor([agree]),
                           torch.tensor([10]), radius_candidates=torch.tensor([radius]))

    def test_domain_disagreement_blocks_only_rescue(self):
        bank = self.bank([.45, .55])  # fused support .725; target disagrees
        self.assertTrue(self.select(bank, high=True, radius=False)[0].item())
        self.assertFalse(self.select(bank)[0].item())

    def test_same_point_seven_threshold_and_self_exclusion(self):
        bank = self.bank([.4, .6])  # high path passes .7, target disagrees
        self.assertTrue(self.select(bank, high=True)[0].item())
        self.assertFalse(self.select(bank)[0].item())
        bank = self.bank([.6, .4])  # support .8, BOTH agree
        self.assertTrue(self.select(bank)[0].item())
        self.assertAlmostEqual(self.select(bank)[4].item(), .6, places=5)

    def test_agreement_and_candidate_gates(self):
        bank = self.bank([.6, .4])
        self.assertFalse(self.select(bank, agree=False)[0].item())
        self.assertFalse(self.select(bank, radius=False)[0].item())

    def test_legacy_confidence_call_unchanged(self):
        bank = self.bank([.45, .55])
        out = bank.select(torch.tensor([[1., 0.]]), torch.tensor([0]),
                          torch.tensor([True]), torch.tensor([True]), torch.tensor([10]))
        self.assertTrue(out[0].item())


if __name__ == '__main__':
    unittest.main()
