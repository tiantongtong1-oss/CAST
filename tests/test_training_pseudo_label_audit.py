import unittest

import numpy as np

from training_pseudo_label_audit import TrainingPseudoLabelAudit


class TrainingPseudoLabelAuditTests(unittest.TestCase):
    def test_accumulates_the_actual_masks_across_batches(self):
        audit = TrainingPseudoLabelAudit()
        audit.update(
            np.array([0, 1, 6, 4]), np.array([0, 2, 6, 4]),
            np.array([True, True, True, False]),
            np.array([True, False, False, False]),
            np.array([False, True, True, False]),
            np.array([True, True, False, False]),
        )
        audit.update(
            np.array([6, 4]), np.array([6, 4]),
            np.array([True, True]), np.array([False, True]),
            np.array([True, False]), np.array([True, False]),
        )

        self.assertEqual(audit.counts['before'].sum(), 6)
        self.assertEqual(audit.counts['radius_candidates'].sum(), 3)
        self.assertEqual(audit.counts['radius_final'].sum(), 2)
        self.assertEqual(audit.correct['radius_final'].sum(), 1)
        self.assertEqual(audit.counts['final'].sum(), 3)
        self.assertEqual(audit.correct['final'].sum(), 2)
        report = audit.markdown(19)
        self.assertIn('第 20 轮', report)
        self.assertIn('| 最终接纳 | 3 | 2 | 66.67% | 50.00% |', report)
        self.assertIn('| 积极 | 1 | 1 | 0 | 100.00% |', report)
        self.assertIn('| 消极 | 1 | 0 | 1 | 0.00% |', report)
        self.assertIn('| 合计 | 2 | 1 | 1 | 50.00% |', report)
        self.assertEqual(report.count('| --- |'), 3)

    def test_no_rescue_reports_undefined_accuracy(self):
        audit = TrainingPseudoLabelAudit()
        audit.update(
            np.array([2]), np.array([2]),
            np.array([True]), np.array([True]),
            np.array([False]), np.array([True]),
        )
        self.assertIn('| 合计 | 0 | 0 | 0 | — |', audit.markdown(0))


if __name__ == '__main__':
    unittest.main()
