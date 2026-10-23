import unittest
import numpy as np
from feature_knn_diagnostics import predict_from_source,compare_predictions

class FeatureKNNTests(unittest.TestCase):
    def test_known_classes_and_scale_invariance(self):
        source=np.repeat(np.eye(7),11,axis=0);labels=np.repeat(np.arange(7),11)
        r=predict_from_source(source,labels,np.eye(7),chunk_size=2)
        np.testing.assert_array_equal(r['knn'],np.arange(7))
        np.testing.assert_array_equal(r['classwise'],np.arange(7))
        np.testing.assert_array_equal(r['votes'].sum(1),np.full(7,11))
        scaled=predict_from_source(source*4,labels,np.eye(7)*9)
        np.testing.assert_array_equal(r['knn'],scaled['knn'])

    def test_classwise_uses_same_k_and_euclidean_mean(self):
        source=np.repeat(np.eye(7),3,axis=0);labels=np.repeat(np.arange(7),3)
        r=predict_from_source(source,labels,[[1,0,0,0,0,0,0]],k=2)
        self.assertAlmostEqual(float(r['classwise_mean_distance'][0,0]),0)
        self.assertAlmostEqual(float(r['classwise_mean_distance'][0,1]),2**.5,places=6)

    def test_recovery_precision_and_losses(self):
        s=compare_predictions([4,1,6,4],[1,4,1,1],[1,1,6,4],[1,1,4,4])
        m=s['source_knn']['fear_vs_head']
        self.assertEqual(m['recovered_true_fear'],1)
        self.assertEqual(m['lost_head_correct_fear'],1)
        self.assertAlmostEqual(m['new_fear_precision'],1/3)
        self.assertEqual(m['net_correct_fear_change'],0)

    def test_missing_reference_and_invalid_features(self):
        source=np.repeat(np.eye(7),2,axis=0);labels=np.repeat(np.arange(7),2)
        with self.assertRaises(ValueError):predict_from_source(source,labels,np.eye(7),k=3)
        with self.assertRaises(ValueError):predict_from_source(source,labels,np.zeros((1,7)),k=2)

if __name__=='__main__':unittest.main()
