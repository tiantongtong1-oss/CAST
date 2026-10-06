import ast
import importlib.util
from pathlib import Path
import unittest
import numpy as np


class SplitTests(unittest.TestCase):
    def test_stratified_disjoint_reproducible(self):
        # Exercise split without importing the GPU training entry point.
        tree = ast.parse(Path('train_source_b.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'stratified_split')
        ns = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<split>', 'exec'), ns)
        labels = np.repeat(np.arange(7), [10,11,12,13,14,15,16])
        train, val = ns['stratified_split'](labels, .2, 2000)
        self.assertFalse(set(train)&set(val))
        self.assertEqual(sorted(train+val), list(range(len(labels))))
        self.assertEqual(ns['stratified_split'](labels, .2, 2000), (train,val))
        self.assertEqual(set(labels[val]), set(range(7)))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required for margin gradient check')
class MarginTests(unittest.TestCase):
    def test_hard_negative_and_gradient(self):
        import torch
        from source_margin import prototype_margin_loss
        p = torch.eye(3, requires_grad=True)
        z = torch.tensor([[0.,1.,0.]], requires_grad=True)
        loss = prototype_margin_loss(z, torch.tensor([0]), p, .1)
        self.assertAlmostEqual(loss.item(), 1.1, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertIsNone(p.grad)
        self.assertEqual(prototype_margin_loss(torch.eye(3), torch.arange(3), p, .1).item(), 0)
