import unittest
import torch
from prototype_radius import PrototypeRadiusBank


class ORRescueTests(unittest.TestCase):
    def bank(self):
        bank = PrototypeRadiusBank(torch.eye(2), torch.ones(2, dtype=torch.bool))
        features = torch.tensor([[1., 0.]] * 7 + [[0., 1.]] * 7)
        bank.set_source_memory(features, torch.tensor([0] * 7 + [1] * 7))
        return bank

    def test_high_confidence_bypasses_radius_but_not_agreement(self):
        bank = self.bank()
        final, radius, neighbor, checked = bank.select(
            torch.tensor([[0., 1.], [1., 0.]]), torch.tensor([0, 0]),
            torch.tensor([True, True]), torch.tensor([True, False]), False)
        self.assertEqual(final.tolist(), [True, False])
        self.assertFalse((radius | neighbor | checked).any())

    def test_radius_then_neighbor_and_warmup(self):
        bank = self.bank()
        bank.radii[0] = 1.
        features = torch.tensor([[1., 0.], [0., 1.], [0., 1.]])
        labels = torch.tensor([0, 1, 0])
        confidence = torch.zeros(3, dtype=torch.bool)
        agreement = torch.ones(3, dtype=torch.bool)
        final, radius, neighbor, checked = bank.select(features, labels, confidence, agreement, True)
        self.assertEqual(radius.tolist(), [True, False, False])
        self.assertEqual(neighbor.tolist(), [False, True, False])
        self.assertEqual(checked.tolist(), [False, True, True])
        self.assertEqual(final.tolist(), [True, True, False])
        self.assertFalse(bank.select(features, labels, confidence, agreement, False)[0].any())

    def test_missing_neighbors_and_refresh(self):
        bank = self.bank()
        bank.set_source_memory(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))
        passed, checked = bank.neighbor_accepts(torch.tensor([[1., 0.]]), torch.tensor([0]), torch.tensor([True]))
        self.assertFalse(passed.any())
        self.assertFalse(checked.any())
        self.assertEqual(bank.counts.tolist(), [1, 1])
        self.assertFalse(bank.ready.any())


if __name__ == '__main__':
    unittest.main()
