import unittest
import torch
from retrain_balanced_head import metrics, stratified_split, frozen_digest


class BalancedHeadTests(unittest.TestCase):
    def test_split_disjoint_stratified_repeatable(self):
        labels = [c for c in range(7) for _ in range(c+3)]
        train, val = stratified_split(labels, .2, 1314)
        self.assertFalse(set(train)&set(val))
        self.assertEqual(set(train+val), set(range(len(labels))))
        self.assertEqual((train,val),stratified_split(labels,.2,1314))
        self.assertEqual({labels[i] for i in val},set(range(7)))

    def test_frozen_bn_but_head_learns(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4),torch.nn.Dropout(.5))
                self.fc=torch.nn.Linear(4,7,bias=False)
                self.bn=torch.nn.BatchNorm1d(7)
        m=Toy().eval()
        for p in m.parameters(): p.requires_grad_(False)
        for p in m.fc.parameters(): p.requires_grad_(True)
        frozen=frozen_digest(m); before=m.fc.weight.detach().clone()
        opt=torch.optim.SGD(m.fc.parameters(),lr=.1)
        with torch.no_grad(): x=m.feature(torch.randn(8,4))
        torch.nn.functional.cross_entropy(m.bn(m.fc(x)),torch.arange(8)%7).backward()
        opt.step()
        self.assertEqual(frozen,frozen_digest(m))
        self.assertFalse(torch.equal(before,m.fc.weight))
        self.assertTrue(all(p.grad is None for p in m.feature.parameters()))

    def test_metrics_include_missing_classes(self):
        r=metrics(torch.tensor([0,0,1]),torch.tensor([0,1,1]))
        self.assertAlmostEqual(r['accuracy'],2/3)
        self.assertAlmostEqual(r['recall'][0],.5)
        self.assertAlmostEqual(r['precision'][1],.5)


if __name__=='__main__': unittest.main()
