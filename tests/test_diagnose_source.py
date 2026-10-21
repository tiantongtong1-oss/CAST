import unittest
import numpy as np
from diagnose_source import class_metrics

class SourceMetricTests(unittest.TestCase):
    def test_confusion_orientation_and_absent_classes(self):
        matrix = np.zeros((7,7), dtype=int)
        matrix[1,1]=2
        matrix[1,4]=6
        matrix[4,1]=3
        matrix[4,4]=9
        rows, uar, macro = class_metrics(matrix)
        self.assertEqual(rows[1]['support'],8)
        self.assertEqual(rows[1]['predicted'],5)
        self.assertAlmostEqual(rows[1]['recall'],.25)
        self.assertAlmostEqual(rows[1]['precision'],.4)
        self.assertAlmostEqual(uar,.5)
        self.assertTrue(np.isfinite(macro))
