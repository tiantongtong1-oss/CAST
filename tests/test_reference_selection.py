import unittest
from diagnose_reference_selection import reference_scores, classify_path

class ReferenceTests(unittest.TestCase):
    def test_reference_can_disagree_without_changing_selection(self):
        score = reference_scores([.01,.95,.01,0,.02,.01,0],4,True,.92)
        self.assertTrue(score['reference_contradiction'])
        self.assertFalse(score['reference_pass'])
        self.assertEqual(score['reference_top'],'Fear')
        self.assertAlmostEqual(score['reference_fear_margin'],.93)
        self.assertEqual(classify_path(True,True,False,False,0,.92),'direct')
    def test_low_support_is_not_contradiction(self):
        score = reference_scores([.1,.2,.1,.1,.3,.1,.1],4,True,.92)
        self.assertFalse(score['reference_pass'])
        self.assertFalse(score['reference_contradiction'])
    def test_unavailable_cannot_pass(self):
        score = reference_scores([0,1,0,0,0,0,0],1,False,.92)
        self.assertFalse(score['reference_pass'])
        self.assertFalse(score['reference_contradiction'])
        self.assertEqual(score['reference_top'],'unavailable')
