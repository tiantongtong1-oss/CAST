import unittest
import numpy as np
from source_sampling import mild_sampling_weights

class SamplingTests(unittest.TestCase):
    def test_sqrt_class_mass_and_epoch_size(self):
        labels=np.repeat(np.arange(7),[1290,281,717,4772,1982,705,2524])
        weights,counts,expected=mild_sampling_weights(labels)
        self.assertAlmostEqual(expected.sum(),len(labels))
        self.assertAlmostEqual(expected[3]/expected[1],(4772/281)**.5)
        self.assertGreater(expected[1],counts[1])
        self.assertAlmostEqual(weights[labels==1][0]/weights[labels==3][0],(4772/281)**.5)

    def test_uniform_and_full_balance_endpoints(self):
        labels=np.array([0,0,0,1])
        np.testing.assert_allclose(mild_sampling_weights(labels,2,0)[2],[3,1])
        np.testing.assert_allclose(mild_sampling_weights(labels,2,1)[2],[2,2])

    def test_torch_sampler_size_and_reproducibility(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable')
        labels=np.array([0,0,0,1])
        w,_,_=mild_sampling_weights(labels,2)
        def draw():
            sampler=torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(w,dtype=torch.double),len(labels),replacement=True,
                generator=torch.Generator().manual_seed(1341))
            return list(sampler)
        self.assertEqual(len(draw()),len(labels))
        self.assertEqual(draw(),draw())

    def test_missing_class_and_invalid(self):
        self.assertEqual(mild_sampling_weights(np.array([0,0]),3)[2][1],0)
        for labels in [[],[1.5],[-1],[7]]:
            with self.assertRaises(ValueError): mild_sampling_weights(labels)
        for power in [-1,np.nan,2]:
            with self.assertRaises(ValueError): mild_sampling_weights(np.array([0]),power=power)

if __name__=='__main__':unittest.main()
