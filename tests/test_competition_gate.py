import unittest
import numpy as np
from competition_gate import competition_audit, supervision_mask

class CompetitionTests(unittest.TestCase):
    def audit(self, votes=None, prob=None, features=None, ready=None, available=True):
        return competition_audit(
            np.array([prob if prob is not None else [.8,.1,.1]]),
            np.array([[.8,.1,.1]]),
            np.array([features if features is not None else [1.,0.]]),
            [0], np.array([[1.,0.],[0.,1.],[-1.,0.]]),
            ready if ready is not None else [True]*3,
            np.array([votes if votes is not None else [7,3,1]]), [available])

    def test_boundary_votes_rejected_and_clear_votes_pass(self):
        self.assertFalse(self.audit(votes=[6,5,0])['pass_mask'][0])
        self.assertTrue(self.audit(votes=[7,4,0])['pass_mask'][0])
        self.assertTrue(self.audit(votes=[6,3,2])['pass_mask'][0])

    def test_both_views_probability_margin_required(self):
        self.assertFalse(self.audit(prob=[.48,.45,.07])['pass_mask'][0])
        self.assertTrue(self.audit()['pass_mask'][0])

    def test_competing_prototype_rejects_wrong_or_ambiguous_class(self):
        self.assertFalse(self.audit(features=[0.,1.])['pass_mask'][0])
        self.assertFalse(self.audit(features=[.70710678,.70710678])['pass_mask'][0])

    def test_invalid_or_unavailable_evidence(self):
        self.assertFalse(self.audit(available=False)['pass_mask'][0])
        self.assertFalse(self.audit(ready=[False,True,True])['pass_mask'][0])
        self.assertFalse(self.audit(features=[0.,0.])['pass_mask'][0])
        self.assertFalse(self.audit(features=[np.nan,0.])['pass_mask'][0])

    def test_pending_rescue_cannot_supervise(self):
        confidence=np.array([True,False,False,False])
        rescue=np.array([False,True,True,False])
        anchors=np.array([False,False,True,False])
        np.testing.assert_array_equal(supervision_mask(confidence,rescue,anchors),[True,False,True,False])

    def test_no_competitor_fails_closed(self):
        self.assertFalse(self.audit(ready=[True,False,False])['pass_mask'][0])

if __name__=='__main__': unittest.main()
