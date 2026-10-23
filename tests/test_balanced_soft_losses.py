import unittest
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'PyTorch unavailable; run on training host')
class BalancedSoftLossTests(unittest.TestCase):
    def test_active_class_average_not_frequency_weighted(self):
        from balanced_soft_losses import equal_class_alignment
        source=[torch.zeros(2,2),torch.zeros(20,2)]
        target=[torch.ones(2,2),torch.full((30,2),3.)]
        fn=lambda a,b:(a.mean(0)-b.mean(0)).square().mean()
        loss,active=equal_class_alignment(source,target,fn)
        self.assertEqual(active,[0,1]);self.assertAlmostEqual(loss.item(),5)
        same,_=equal_class_alignment(source,[target[0].repeat(10,1),target[1]],fn)
        self.assertAlmostEqual(same.item(),5)

    def test_sparse_class_skipped_and_zero_has_backward(self):
        from balanced_soft_losses import equal_class_alignment
        a=torch.randn(1,2,requires_grad=True);b=torch.randn(2,2,requires_grad=True)
        loss,active=equal_class_alignment([a],[b],lambda x,y:x.sum())
        self.assertEqual(active,[]);self.assertEqual(loss.item(),0)
        loss.backward();self.assertIsNotNone(a.grad)

    def test_pending_no_gradient_teacher_detached_soft_gradient_exact(self):
        from balanced_soft_losses import classification_with_soft_rescue
        logits=torch.zeros(4,3,requires_grad=True)
        teacher=torch.tensor([[.1,.8,.1],[.2,.3,.5],[.6,.2,.2]],requires_grad=True)
        loss,_=classification_with_soft_rescue(logits,torch.tensor([0]),torch.tensor([1,1,0]),
            torch.tensor([True,False,False]),torch.tensor([False,False,True]),teacher,.25)
        loss.backward()
        torch.testing.assert_allclose(logits.grad[2],torch.zeros(3))
        torch.testing.assert_allclose(logits.grad[3],.25*(torch.full((3,),1/3)-teacher.detach()[2])/3)
        self.assertIsNone(teacher.grad)

    def test_zero_soft_weight_and_unit_onehot_matches_hard(self):
        from balanced_soft_losses import classification_with_soft_rescue
        logits=torch.randn(3,2,requires_grad=True)
        q=torch.tensor([[1.,0.],[0.,1.]])
        loss,_=classification_with_soft_rescue(logits,torch.tensor([0]),torch.tensor([0,1]),
            torch.tensor([True,False]),torch.tensor([False,True]),q,1.)
        ref=torch.nn.functional.cross_entropy(logits,torch.tensor([0,0,1]))
        torch.testing.assert_allclose(loss,ref)
        zero,_=classification_with_soft_rescue(logits,torch.tensor([0]),torch.tensor([0,1]),
            torch.tensor([True,False]),torch.tensor([False,True]),q,0.)
        g=torch.autograd.grad(zero,logits)[0]
        torch.testing.assert_allclose(g[2],torch.zeros(2))

    def test_overlapping_soft_mask_cannot_double_supervise(self):
        from balanced_soft_losses import classification_with_soft_rescue
        z=torch.zeros(2,2,requires_grad=True)
        loss,_=classification_with_soft_rescue(z,torch.tensor([0]),torch.tensor([1]),
            torch.tensor([True]),torch.tensor([True]),torch.tensor([[1.,0.]]),.25)
        torch.testing.assert_allclose(loss,torch.nn.functional.cross_entropy(z,torch.tensor([0,1])))

if __name__=='__main__':unittest.main()
