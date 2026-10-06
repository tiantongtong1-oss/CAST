import importlib.util
import unittest
import numpy as np
from multi_prototypes import class_multi_prototypes


class ClusteringTests(unittest.TestCase):
    def test_modes_reproducible_and_counts(self):
        x = np.tile(np.array([[1.,0.],[.99,.01],[-1.,0.],[-.99,.01]]), (7,1))
        y = np.repeat(np.arange(7),4)
        centers, counts = class_multi_prototypes(x,y,k=2)
        again, _ = class_multi_prototypes(x,y,k=2)
        np.testing.assert_array_equal(centers, again)
        np.testing.assert_allclose(np.linalg.norm(centers,axis=-1),1,atol=1e-6)
        self.assertEqual(counts, [[2,2]]*7)
        self.assertLess(float(centers[0,0]@centers[0,1]),-.9)

    def test_k1_equals_B_mean(self):
        x = np.tile(np.array([[1.,0.],[1.,1.]]), (7,1))
        y = np.repeat(np.arange(7),2)
        centers, _ = class_multi_prototypes(x,y,k=1)
        unit = x[:2]/np.linalg.norm(x[:2],axis=1,keepdims=True)
        mean = unit.mean(0); mean /= np.linalg.norm(mean)
        np.testing.assert_allclose(centers[0,0],mean,atol=1e-6)

    def test_insufficient_class_members(self):
        with self.assertRaises(ValueError):
            class_multi_prototypes(np.eye(7),np.arange(7),k=3)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class MarginTests(unittest.TestCase):
    def test_k1_equivalence_and_detached_centers(self):
        import torch
        from source_margin import prototype_margin_loss
        from multi_source_margin import multi_prototype_margin_loss
        centers = torch.eye(3,requires_grad=True)
        z = torch.tensor([[0.,1.,0.]],requires_grad=True)
        labels = torch.tensor([0])
        multi = multi_prototype_margin_loss(z,labels,centers[:,None],.1)
        self.assertTrue(torch.allclose(multi,prototype_margin_loss(z,labels,centers,.1)))
        multi.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertIsNone(centers.grad)
