import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from diagnose_fear import summarize, write_reports


class FearDiagnosisTests(unittest.TestCase):
    def test_ranks_confusion_and_metrics(self):
        p = np.zeros((4,7))
        p[0,:3] = [.1,.7,.2]
        p[1,:3] = [.6,.3,.1]
        p[2,:3] = [.4,.1,.5]
        p[3,:3] = [.2,.6,.2]
        matrix, pred, ranks, gap, stats = summarize([1,1,1,0], p)
        np.testing.assert_array_equal(ranks, [1,2,3,1])
        np.testing.assert_array_equal(matrix[1,:3], [1,1,1])
        self.assertAlmostEqual(stats['fear_recall'], 1/3)
        self.assertAlmostEqual(stats['fear_precision'], .5)
        self.assertAlmostEqual(stats['fear_top2_recall'], 2/3)
        self.assertAlmostEqual(gap[1], .3)
        with tempfile.TemporaryDirectory() as tmp:
            write_reports(tmp, ['a','b','c','d'], np.array([1,1,1,0]), p, {})
            self.assertEqual(len((Path(tmp)/'fear_predictions.csv').read_text().splitlines()), 4)
            self.assertEqual(json.loads((Path(tmp)/'summary.json').read_text())['fear_support'], 3)

    def test_ties_and_absent_fear(self):
        _, pred, ranks, _, stats = summarize([0], np.ones((1,7))/7)
        self.assertEqual(pred[0], 0)
        self.assertEqual(ranks[0], 2)
        self.assertIsNone(stats['fear_top2_recall'])

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            summarize([1], np.ones((1,7)))
