import unittest
import numpy as np
from adaptive_distance_vote import AdaptiveDistanceVoteBank, audit_adaptive_vote
from dual_domain_ds import DSConfig

class AdaptiveVoteTests(unittest.TestCase):
    def audit(self, votes):
        v=np.array([votes]); n=v.sum(1)
        s=dict(votes=v,count=n,mass=v/np.maximum(n[:,None],1),available=n>=3,uncertainty=np.zeros(1))
        return bool(audit_adaptive_vote(s,[0])['pass_mask'][0])

    def test_actual_count_rules(self):
        for votes,expected in [([2,0],False),([3,0],True),([2,1],False),
            ([4,0],True),([4,1],False),([5,0],True),([5,1],True),
            ([5,2],False),([6,2],True),([6,5],False),([8,3],False),([9,2],True)]:
            with self.subTest(votes=votes): self.assertEqual(self.audit(votes),expected)

    def test_calibration_excludes_self(self):
        bank=AdaptiveDistanceVoteBank([[1,0],[0,1],[-1,0]],[0,0,1],[0,1,2],[1]*3,2)
        thresholds=bank.calibrate()
        self.assertAlmostEqual(thresholds[0],2**.5,places=6)
        self.assertTrue(np.isnan(thresholds[1]))

    def test_far_neighbors_removed_and_short_bank_allowed(self):
        bank=AdaptiveDistanceVoteBank([[1,0]]*3+[[-1,0]]*8,[0]*3+[1]*8,list(range(11)),[1]*11,2)
        bank.calibrate()
        r=bank.evaluate([[1,0]],[99],DSConfig(k=11),[0])
        self.assertEqual(r['raw_count'][0],11)
        self.assertEqual(r['count'][0],3)
        self.assertTrue(audit_adaptive_vote(r,[0])['pass_mask'][0])
        short=AdaptiveDistanceVoteBank([[1,0]]*3,[0]*3,list(range(3)),[1]*3,2)
        short.calibrate()
        r=short.evaluate([[1,0]],[99],DSConfig(k=11),[0])
        self.assertTrue(audit_adaptive_vote(r,[0])['pass_mask'][0])

    def test_missing_class_invalid_query_and_zero_neighbors(self):
        bank=AdaptiveDistanceVoteBank([[1,0]]*3,[0]*3,list(range(3)),[1]*3,2)
        bank.calibrate()
        r=bank.evaluate([[0,1],[0,0],[np.nan,0],[1,0]],[99]*4,DSConfig(k=11),[0,0,0,1])
        self.assertFalse(r['available'].any())
        self.assertFalse(audit_adaptive_vote(r,[0,0,0,1])['pass_mask'].any())

    def test_maximum_eleven_and_exclude_query(self):
        bank=AdaptiveDistanceVoteBank([[1,0]]*14,[0]*14,list(range(14)),[1]*14,2)
        bank.calibrate()
        r=bank.evaluate([[1,0]],[0],DSConfig(k=11),[0],exclude_self=True)
        self.assertEqual(r['count'][0],11)

if __name__=='__main__': unittest.main()
