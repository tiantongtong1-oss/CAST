import itertools
import unittest
import numpy as np
from competition_gate import promotion_eligibility, supervision_mask


class PromotionOrTests(unittest.TestCase):
    def test_all_evidence_combinations_and_agreement_veto(self):
        rows = np.array(list(itertools.product([False, True], repeat=4)))
        agreement, confidence, radius, neighbor = rows.T
        actual = promotion_eligibility(agreement, confidence, radius, neighbor)
        expected = np.array([a and any((c, r, n)) for a, c, r, n in rows])
        np.testing.assert_array_equal(actual, expected)

    def test_new_anchors_do_not_bypass_training_acceptance(self):
        # Radius/neighbor can now promote even when competition rejects rescue.
        agree = np.ones(4, dtype=bool)
        confidence = np.array([True, False, False, False])
        radius = np.array([False, True, False, False])
        neighbor = np.array([False, False, True, False])
        anchors = promotion_eligibility(agree, confidence, radius, neighbor)
        np.testing.assert_array_equal(anchors, [True, True, True, False])
        rescue = np.array([False, False, False, False])
        np.testing.assert_array_equal(supervision_mask(confidence, rescue, anchors),
                                      [True, False, False, False])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class PromotionHistoryTests(unittest.TestCase):
    def test_consecutive_or_routes_label_change_and_revocation(self):
        from reliability_roles import ReliabilityRoles
        tracker = ReliabilityRoles(1, 3)
        ids = torch.tensor([0])
        label = torch.tensor([1])
        yes, no = torch.tensor([True]), torch.tensor([False])
        quality = torch.tensor([.8])
        # Alternate qualifying evidence routes; same label promotes on epoch 2.
        for epoch, route in enumerate(((yes, no, no), (no, yes, no), (no, no, yes))):
            eligible = promotion_eligibility(yes, *route)
            anchor = tracker.observe(ids, label, eligible, quality, epoch)
            self.assertEqual(anchor.item(), epoch == 2)
        self.assertFalse(tracker.observe(ids, torch.tensor([2]), yes, quality, 3).item())
        self.assertEqual(tracker.streak.item(), 1)
        self.assertFalse(tracker.observe(ids, torch.tensor([2]), no, quality, 4).item())
        self.assertEqual(tracker.streak.item(), 0)


if __name__ == '__main__':
    unittest.main()
