import unittest
import torch
from global_gaussian_knn_reliability import GlobalGaussianKNNReliabilityBank

class RescueTests(unittest.TestCase):
    def make_bank(self):
        return GlobalGaussianKNNReliabilityBank(3, 2, k=2, num_target_samples=4)

    def test_candidate_margin_and_unrelated_class(self):
        bank = self.make_bank()
        def fake_gate(features, labels, ids, active, thresholds):
            # Unrelated class 2 has strongest support but must not be selected.
            support = features.new_full((features.shape[0],), [0.9, 0.3, 1.0][int(labels[0])])
            return {"score": support, "weighted_support": support,
                    "pass_mask": torch.ones_like(active)}
        bank.gate = fake_gate
        args = (torch.ones(1, 2), torch.tensor([0]), torch.ones(3)*.8,
                torch.tensor([[0,1]]))
        result = bank.query_class_support(*args, support_margin=.2)
        self.assertEqual(result["labels"].item(), 0)
        self.assertTrue(result["reliable"].item())
        self.assertFalse(bank.query_class_support(*args, support_margin=.7)["reliable"].item())

    def test_quarantine_survives_refresh_and_checkpoint(self):
        bank = self.make_bank()
        bank.quarantine(torch.tensor([1]))
        bank.begin_source_refresh()
        bank.set_target_memory(torch.tensor([[1.,0.],[1.,0.],[1.,0.]]),
                               torch.zeros(3, dtype=torch.long),
                               torch.ones(3), torch.tensor([0,1,2]))
        self.assertTrue(bank.is_quarantined(torch.tensor([1])).item())
        restored = self.make_bank()
        restored.load_state_dict(bank.state_dict())
        self.assertTrue(restored.is_quarantined(torch.tensor([1])).item())

    def test_quarantined_neighbor_cannot_support(self):
        bank = self.make_bank()
        bank.distribution_initialized[:] = True
        bank.sigma_initialized.fill_(True)
        bank.global_var.fill_(1.)
        bank.class_means[:] = torch.tensor([1.,0.])
        bank.set_target_memory(torch.tensor([[1.,0.],[1.,0.]]),
                               torch.zeros(2, dtype=torch.long),
                               torch.ones(2), torch.tensor([1,2]))
        args = (torch.tensor([[1.,0.]]), torch.tensor([0]), torch.tensor([0]),
                torch.tensor([True]), torch.ones(3)*.8)
        self.assertTrue(bank.gate(*args)["pass_mask"].item())
        bank.quarantine(torch.tensor([1,2]))
        result = bank.gate(*args)
        self.assertEqual(result["weighted_support"].item(), 0.)
        self.assertFalse(result["pass_mask"].item())

if __name__ == "__main__":
    unittest.main()
