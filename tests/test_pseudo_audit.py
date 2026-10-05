import unittest
import numpy as np
from pseudo_audit import PredictionAudit, target_training_mask


class MaskArray:
    def __init__(self, values):
        self.values = np.asarray(values)
    def clone(self):
        return MaskArray(self.values.copy())
    def ne(self, value):
        return MaskArray(self.values != value)
    def __and__(self, other):
        return MaskArray(self.values & other.values)


class AuditTests(unittest.TestCase):
    def test_confusion_orientation_and_accumulation(self):
        audit = PredictionAudit()
        audit.add([5, 2, 3], [2, 2, 3], {'confidence_AND_KNN': [True, False, True]})
        audit.add([2], [5], {'confidence_AND_KNN': [True]})
        self.assertEqual(audit.matrices['all'][5, 2], 1)
        self.assertEqual(audit.matrices['all'][2, 5], 1)
        self.assertEqual(audit.matrices['all'][2, 2], 1)
        self.assertEqual(audit.matrices['confidence_AND_KNN'][2, 2], 0)
        self.assertEqual(audit.total, 4)
        self.assertEqual(audit.true_counts[2], 2)

    def test_empty_groups(self):
        audit = PredictionAudit()
        audit.add([2], [2], {'training': [False]})
        self.assertEqual(audit.matrices['training'].sum(), 0)
        audit.report('empty group')

    def test_exclusion_preserves_selection_and_default(self):
        selected = MaskArray([True, True, False, True])
        labels = MaskArray([2, 5, 3, 2])
        np.testing.assert_array_equal(target_training_mask(selected, labels).values,
                                      selected.values)
        np.testing.assert_array_equal(target_training_mask(selected, labels, True).values,
                                      [False, True, False, False])
        np.testing.assert_array_equal(selected.values, [True, True, False, True])

    def test_invalid_ids(self):
        with self.assertRaises(ValueError):
            PredictionAudit().add([7], [0])


if __name__ == '__main__':
    unittest.main()
