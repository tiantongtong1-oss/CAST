import unittest
import numpy as np
from class_neighbor_diagnostics import class_neighbor_scores


class ClassTests(unittest.TestCase):
    def test_competitor_not_global_topk(self):
        f = [[1,0],[1,0],[0,1],[0,1]]
        r = class_neighbor_scores([[1,0]], [0], f, [0,0,1,1], k=2, num_classes=2)
        self.assertTrue(r['ready'][0])
        self.assertAlmostEqual(r['margin'][0], 1)
        self.assertEqual(r['competitor'][0], 1)

    def test_self_exclusion_and_insufficient(self):
        r = class_neighbor_scores([[1,0]], [0], [[1,0],[0,1]], [0,1],
                                  k=1, num_classes=2, query_ids=[0], bank_ids=[0,1])
        self.assertFalse(r['ready'][0]); self.assertTrue(np.isnan(r['margin'][0]))
        self.assertEqual(r['counts'][0,0], 0)

    def test_invalid_and_empty(self):
        r = class_neighbor_scores([[1,0],[0,0],[float('nan'),0]], [0,0,0],
                                  np.empty((0,2)), [], k=1, num_classes=2)
        self.assertFalse(r['ready'].any())
        with self.assertRaises(ValueError):
            class_neighbor_scores([[1,0]], [0], [[1,0]], [0], query_ids=[0])

    def test_bruteforce(self):
        rng = np.random.default_rng(4)
        q = rng.normal(size=(5,3)); f=rng.normal(size=(18,3)); labels=np.repeat(np.arange(3),6)
        r=class_neighbor_scores(q, np.zeros(5,int), f, labels, k=3,num_classes=3)
        sim=(q/np.linalg.norm(q,axis=1,keepdims=True)) @ (f/np.linalg.norm(f,axis=1,keepdims=True)).T
        for c in range(3):
            np.testing.assert_allclose(r['scores'][:,c], np.sort(sim[:,labels==c],axis=1)[:,-3:].mean(1),atol=1e-6)


if __name__ == '__main__':
    unittest.main()
