import unittest
import numpy as np
from prototype_lifecycle import PrototypeLifecycle, mutual_components, margins


class LifecycleTests(unittest.TestCase):
    def data(self):
        ids=np.arange(20);z=np.vstack([np.tile([.6,.8],(10,1)),np.tile([.6,-.8],(10,1))]);y=np.zeros(20,int)
        return ids,z,y,np.array([[1.,0.],[-1.,0.]])
    def bank(self):
        return PrototypeLifecycle(classes=2,k=3,min_size=3,persistence=2,
                                  audit_min=2,graph_limit=100)
    def test_geometry_components(self):
        _,z,_,_=self.data()
        groups=mutual_components(z,k=3,min_size=3)
        self.assertEqual(len(groups),2)
    def test_activation_and_disappearance(self):
        ids,z,y,base=self.data();b=self.bank()
        first=b.update(ids,z,y,base,base,0);self.assertEqual(first['active'],0)
        second=b.update(ids,z,y,base,base,1);self.assertEqual(second['active'],2)
        self.assertTrue(all(all(i%5!=0 for i in t['members']) for t in b.tracks))
        before=margins(z,y,base,[]);after=margins(z,y,base,b.active)
        self.assertTrue(np.all(after>=before))
        b.update([],np.empty((0,2)),[],base,base,2);self.assertEqual(len(b.active),0)
    def test_external_no_gain_reject(self):
        ids,z,y,base=self.data();z[ids%5==0]=[1,0];b=self.bank()
        b.update(ids,z,y,base,base,0);r=b.update(ids,z,y,base,base,1)
        self.assertEqual(r['active'],0);self.assertGreater(r['audit_failed'],0)
    def test_source_semantic_reject(self):
        ids,z,y,base=self.data();b=self.bank()
        r=b.update(ids,z,y,-base,base,0);self.assertEqual(r['tracked'],0)
    def test_duplicate_reject(self):
        ids,z,y,base=self.data();ids[1]=ids[0]
        with self.assertRaises(ValueError):self.bank().update(ids,z,y,base,base,0)
    def test_audit_member_change_not_centroid(self):
        ids,z,y,base=self.data();other=z.copy();other[ids%5==0]=[1,0]
        a=self.bank();b=self.bank();a.update(ids,z,y,base,base,0);b.update(ids,other,y,base,base,0)
        for x,t in zip(a.tracks,b.tracks):np.testing.assert_allclose(x['prototype'],t['prototype'])


if __name__=='__main__':unittest.main()
