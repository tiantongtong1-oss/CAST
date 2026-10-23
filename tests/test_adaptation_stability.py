import unittest
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'PyTorch unavailable; run on training host')
class StabilityTests(unittest.TestCase):
    def test_freezes_stats_but_keeps_affine_gradients(self):
        from adaptation_stability import freeze_backbone_bn_statistics
        feature=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3))
        feature.train();bn=feature[1];before=bn.running_mean.clone()
        self.assertEqual(freeze_backbone_bn_statistics(feature),1)
        result=feature(torch.randn(8,3)+5).sum();result.backward()
        torch.testing.assert_allclose(bn.running_mean,before)
        self.assertIsNotNone(bn.weight.grad);self.assertTrue(bn.weight.requires_grad)

    def test_weighted_gradient_ratio_and_non_mutation(self):
        from adaptation_stability import shared_gradient_metrics
        p=torch.nn.Parameter(torch.tensor([1.,2.]))
        source=p.square().sum();ssl=2*source
        m=shared_gradient_metrics(source,ssl,[p])
        self.assertAlmostEqual(m['ssl_to_source_grad_ratio'],2,places=5)
        self.assertAlmostEqual(m['grad_cosine'],1,places=5)
        self.assertIsNone(p.grad)
        (source+ssl).backward();torch.testing.assert_allclose(p.grad,6*p.detach())

    def test_zero_ssl_and_opposing_gradients(self):
        from adaptation_stability import shared_gradient_metrics
        p=torch.nn.Parameter(torch.tensor([1.,2.]));loss=p.square().sum()
        m=shared_gradient_metrics(loss,loss*0,[p])
        self.assertEqual(m['weighted_ssl_grad_norm'],0);self.assertIsNone(m['grad_cosine'])
        m=shared_gradient_metrics(loss,-loss,[p])
        self.assertAlmostEqual(m['grad_cosine'],-1,places=5)

if __name__=='__main__':unittest.main()
