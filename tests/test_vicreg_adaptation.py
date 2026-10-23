import ast
import tempfile
import unittest
from pathlib import Path
from vicreg_adaptation_utils import discover_unlabeled_train

class LabelIsolationTests(unittest.TestCase):
    def test_only_train_and_no_class_name_parsing(self):
        with tempfile.TemporaryDirectory(dir='.') as root:
            root=Path(root)
            for split in ['train','val','test']:
                folder=root/split/'not_a_label';folder.mkdir(parents=True)
                (folder/'x.jpg').touch()
            paths=discover_unlabeled_train(root)
            self.assertEqual(paths,[str(root/'train/not_a_label/x.jpg')])

    def test_target_dataset_getitem_returns_only_views(self):
        tree=ast.parse(Path('adapt_target_vicreg.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UnlabeledTwoViews')
        returns=[n for n in ast.walk(cls) if isinstance(n,ast.Return)]
        view_return=next(n for n in returns if isinstance(n.value,ast.Tuple))
        self.assertEqual(len(view_return.value.elts),2)
        names={n.attr for n in ast.walk(cls) if isinstance(n,ast.Attribute)}
        self.assertNotIn('label',names)

try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'PyTorch unavailable; run these tensor/gradient tests on training host')
class VICRegTests(unittest.TestCase):
    def test_variance_penalizes_constant_embeddings(self):
        from vicreg_adaptation_utils import vicreg_loss
        x=torch.zeros(8,4,requires_grad=True)
        loss,terms=vicreg_loss(x,x)
        self.assertGreater(terms['variance'].item(),.9)
        self.assertAlmostEqual(terms['invariance'].item(),0)
        self.assertAlmostEqual(terms['covariance'].item(),0)
        loss.backward();self.assertTrue(torch.isfinite(x.grad).all())

    def test_both_views_have_gradients(self):
        from vicreg_adaptation_utils import vicreg_loss
        torch.manual_seed(1)
        x=torch.randn(8,4,requires_grad=True);y=torch.randn(8,4,requires_grad=True)
        loss,terms=vicreg_loss(x,y);loss.backward()
        self.assertGreater(x.grad.norm().item(),0);self.assertGreater(y.grad.norm().item(),0)
        for term in terms.values():self.assertTrue(torch.isfinite(term))

    def test_covariance_counts_offdiagonal_only_and_batch_guard(self):
        from vicreg_adaptation_utils import vicreg_loss
        x=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
        _,terms=vicreg_loss(x,x)
        self.assertAlmostEqual(terms['covariance'].item(),0,places=6)
        self.assertAlmostEqual(terms['variance'].item(),0,places=6)
        with self.assertRaises(ValueError):vicreg_loss(x[:1],x[:1])

if __name__=='__main__':unittest.main()
