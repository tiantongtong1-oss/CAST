import unittest
import numpy as np
from neighbor_structure import StructureBank


class StructureTests(unittest.TestCase):
    def bank(self, **kwargs):
        b=StructureBank(k=4,min_neighbors=3,similarity=.1,**kwargs)
        angles=np.array([-.3,-.2,-.1,0,.1,.2,.3])
        features=np.stack([np.cos(angles),np.sin(angles)],axis=1)
        b.refresh(np.arange(7),features)
        return b,features
    def test_same_views_and_frozen(self):
        b,z=self.bank();r=b.query([3],z[[3]],z[[3]],[True])
        self.assertEqual(r[0],[0]);self.assertEqual(r[3],[0.0])
        self.assertAlmostEqual(float(r[2][0].sum()),1,places=6)
        self.assertFalse(any(np.allclose(x,z[3]) for x in r[1][0]))
        z[:]=0;self.assertGreater(np.linalg.norm(b.features),0)
    def test_reject_unaccepted(self):
        b,z=self.bank();self.assertEqual(b.query([3],z[[3]],z[[3]],[False])[0],[])
    def test_dual_view_disagreement(self):
        b,z=self.bank(max_js=0)
        self.assertEqual(b.query([3],[[1,0]],[[.8,.6]],[True])[0],[])
    def test_empty_or_invalid(self):
        b=StructureBank();b.refresh([],np.empty((0,2)))
        self.assertEqual(b.query([0],[[1,0]],[[1,0]],[True])[0],[])
        b,z=self.bank();self.assertEqual(b.query([3],[[0,0]],[[0,0]],[True])[0],[])
    def test_duplicate_and_deterministic_cap(self):
        with self.assertRaises(ValueError):StructureBank().refresh([1,1],[[1,0],[1,0]])
        b=StructureBank(capacity=5);ids=np.arange(20);z=np.tile([1,0],(20,1));b.refresh(ids,z)
        a=b.ids.copy();b.refresh(ids[::-1],z[::-1]);np.testing.assert_array_equal(a,b.ids)


if __name__=='__main__':unittest.main()
