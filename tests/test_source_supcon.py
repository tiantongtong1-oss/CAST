import unittest
import torch
from train_source_supcon import supervised_contrastive, BalancedBatchSampler


class SourceContrastTests(unittest.TestCase):
    def test_balanced_sampler(self):
        labels=[c for c in range(7) for _ in range(c+2)]
        s=BalancedBatchSampler(labels,list(range(len(labels))),2,42)
        a=list(s); self.assertEqual(a,list(s))
        for batch in a:
            self.assertEqual([sum(labels[i]==c for i in batch) for c in range(7)],[2]*7)
    def test_finite_grad_and_permutation(self):
        torch.manual_seed(2)
        z=torch.randn(14,8,requires_grad=True); y=torch.arange(7).repeat(2)
        loss=supervised_contrastive(z,y)
        order=torch.randperm(14)
        self.assertTrue(torch.allclose(loss,supervised_contrastive(z[order],y[order]),atol=1e-5))
        loss.backward(); self.assertTrue(torch.isfinite(z.grad).all())
    def test_no_positive(self):
        z=torch.randn(7,8,requires_grad=True)
        loss=supervised_contrastive(z,torch.arange(7));loss.backward()
        self.assertEqual(loss.item(),0)

if __name__=='__main__': unittest.main()
