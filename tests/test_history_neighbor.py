import unittest
import numpy as np
from history_neighbor import HistoryNeighborGate


class GateTests(unittest.TestCase):
    def make(self):
        bank = HistoryNeighborGate(7, 2, window=2, k=20, min_remaining=2,
                                   min_groups=2, min_similarity=0.5, support=0.6)
        ids = np.arange(7)
        features = np.tile([1., 0.], (7, 1))
        for epoch in range(2):
            bank.begin_epoch(epoch)
            labels = np.array([0, 0, 0, 0, 0, 0, 1]) if epoch else np.array([0, 0, 0, 1, 1, 2, 1])
            bank.observe(ids, features, labels, np.ones(7, bool),
                         np.ones(7, bool), np.full(7, .9))
        return bank

    def test_snapshot_and_self_exclusion(self):
        bank = self.make()
        # Before epoch 2, snapshot still lacks two complete epochs.
        self.assertFalse(bank.query([0], [[1., 0.]], [0])[0][0])
        bank.begin_epoch(2)
        ok, score, remaining, groups = bank.query([0], [[1., 0.]], [0])
        self.assertTrue(ok[0])
        self.assertEqual(groups[0], 3)
        self.assertEqual(remaining[0], 4)
        self.assertAlmostEqual(score[0], 2/3, places=6)
        # Same-epoch mutations cannot change frozen query evidence.
        bank.observe([3], [[-1., 0.]], [1], [False], [False], [.1])
        self.assertAlmostEqual(bank.query([0], [[1., 0.]], [0])[1][0], score[0])

    def test_similarity_and_conflict_reject(self):
        bank = self.make(); bank.begin_epoch(2)
        self.assertFalse(bank.query([0], [[-1., 0.]], [0])[0][0])
        self.assertFalse(bank.query([0], [[1., 0.]], [1])[0][0])

    def test_single_history_group_reject(self):
        bank = self.make()
        bank.labels[:] = 0
        bank.begin_epoch(2)
        self.assertFalse(bank.query([0], [[1., 0.]], [0])[0][0])

    def test_diagnostic_details(self):
        bank = self.make(); bank.begin_epoch(2)
        result = bank.query([0], [[1., 0.]], [0], return_details=True)
        self.assertEqual(len(result), 5)
        details = result[-1]
        self.assertAlmostEqual(details['support_before'][0], 5/6, places=6)
        self.assertAlmostEqual(details['support_after'][0], 2/3, places=6)
        self.assertAlmostEqual(details['local_margin'][0], 0)

    def test_invalid_and_duplicate(self):
        with self.assertRaises(ValueError):
            HistoryNeighborGate(2, 2, support=2)
        bank = self.make(); bank.begin_epoch(2)
        bank.observe([0], [[1., 0.]], [0], [True], [True], [.9])
        with self.assertRaises(ValueError):
            bank.observe([0], [[1., 0.]], [0], [True], [True], [.9])


if __name__ == '__main__':
    unittest.main()
