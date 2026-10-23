import unittest
import numpy as np
from topk_diagnostics import summarize

class TopKTests(unittest.TestCase):
    def test_candidate_precision_coverage_and_conditionals(self):
        p=np.array([[.01,.35,.01,.01,.6,.01,.01],
                    [.01,.3,.01,.01,.01,.01,.65],
                    [.01,.3,.01,.01,.65,.01,.01],
                    [.01,.8,.01,.01,.01,.01,.15]])
        s,o,r=summarize(p,[1,1,4,1]);fear=s['per_class'][1]
        self.assertEqual(fear['top1']['correct'],1)
        self.assertAlmostEqual(fear['top1']['coverage'],1/3)
        self.assertEqual(fear['top2']['candidate_count'],4)
        self.assertEqual(fear['top2']['candidate_precision'],.75)
        self.assertEqual(fear['top2']['coverage'],1)
        self.assertEqual(s['fear_confusion_conditionals'][0]['fear_rank2_fraction'],1)
        self.assertEqual(s['confusion_rows_true_columns_predicted'][1][4],1)

    def test_empty_groups_and_stable_ties(self):
        s,o,r=summarize([[1/7]*7],[0])
        self.assertIsNone(s['per_class'][1]['top1']['candidate_precision'])
        self.assertIsNone(s['per_class'][1]['top2']['coverage'])
        self.assertIsNone(s['fear_confusion_conditionals'][0]['fear_rank2_fraction'])
        np.testing.assert_array_equal(o[0],np.arange(7))

    def test_invalid(self):
        for p,y in [([[0]*7],[0]),([[np.nan]*7],[0]),([[1/7]*7],[7])]:
            with self.assertRaises(ValueError):summarize(p,y)

if __name__=='__main__':unittest.main()
