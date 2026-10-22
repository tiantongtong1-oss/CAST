import unittest
import numpy as np
from neighbor_vote import NeighborVoteBank, audit_vote, rescue_union
from dual_domain_ds import DSConfig

class VoteTests(unittest.TestCase):
    def bank(self, labels, features=None):
        n=len(labels)
        if features is None: features=np.tile([1.,0.],(n,1))
        return NeighborVoteBank(features, labels, np.arange(n), np.ones(n), 2)
    def test_exact_six_of_eleven_and_five_fails(self):
        for n,expected in [(6,True),(5,False)]:
            evidence=self.bank([0]*n+[1]*(11-n)).evaluate([[1.,0.]],[99],DSConfig(k=11))
            result=audit_vote(evidence,[0],6)
            self.assertEqual(bool(result['pass_mask'][0]),expected)
            self.assertAlmostEqual(result['support'][0],n/11)
    def test_distance_selects_nearest_not_first_or_weighted(self):
        x=np.array([[-1.,0.]]+[[1.,0.]]*11)
        evidence=self.bank([1]+[0]*6+[1]*5,x).evaluate([[1.,0.]],[99],DSConfig(k=11))
        self.assertEqual(audit_vote(evidence,[0])['vote_count'][0],6)
    def test_incomplete_or_invalid_unavailable(self):
        e=self.bank([0]*10).evaluate([[1.,0.],[0.,0.],[np.nan,1.]],[99,98,97],DSConfig(k=11))
        self.assertFalse(e['available'].any())
        self.assertFalse(audit_vote(e,[0,0,0])['pass_mask'].any())
    def test_or_truth_table(self):
        np.testing.assert_array_equal(rescue_union(np.array([False,False,True,True]),np.array([False,True,False,True])),[False,True,True,True])
