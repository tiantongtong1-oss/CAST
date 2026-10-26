"""Execute the live scanner/selector with a NumPy tensor shim; no GPU/data needed."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class Tensor(np.ndarray):
    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def float(self):
        return tensor(self.astype(np.float32))

    def max(self, dim):
        return tensor(np.asarray(self).max(axis=dim)), tensor(np.asarray(self).argmax(axis=dim))

    def eq(self, other):
        return tensor(self == other)

    def size(self, dim):
        return self.shape[dim]

    def numel(self):
        return np.asarray(self).size

    def mean(self):
        return np.asarray(self).mean()

    def index_select(self, dim, index):
        return tensor(np.take(self, index, axis=dim))


def tensor(values):
    return np.asarray(values).view(Tensor)


def softmax(values, dim):
    values = np.asarray(values)
    exp = np.exp(values - values.max(axis=dim, keepdims=True))
    return tensor(exp / exp.sum(axis=dim, keepdims=True))


def load_function(path, name):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    fake_torch = SimpleNamespace(
        minimum=lambda a, b: tensor(np.minimum(a, b)),
        isfinite=lambda a: tensor(np.isfinite(a)),
        cat=lambda parts: tensor(np.concatenate(parts)),
        quantile=lambda a, q, interpolation: np.quantile(np.asarray(a), q, method=interpolation),
    )
    env = dict(np=np, torch=fake_torch, F=SimpleNamespace(softmax=softmax))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


class Teacher:
    def parameters(self):
        return iter([SimpleNamespace(device='cpu')])

    def eval(self):
        return self

    def __call__(self, images, *args, **kwargs):
        return images, None


def batch(scores1, scores2):
    def logits(scores):
        return tensor(np.log([[s, 1 - s] for s in scores]))
    return logits(scores1), logits(scores2), object()  # Labels must never be read.


class GlobalQuantileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = staticmethod(load_function('train.py', 'calculate_global_quantile_threshold'))
        cls.select = staticmethod(load_function('ema_utils.py', 'select_dual_view_pseudo_labels'))

    def test_higher_uses_pooled_scores_from_unequal_batches(self):
        scores = [.60, .65, .70, .75, .80, .85, .90, .93, .96, .99]
        result = self.scan(Teacher(), [batch(scores[:1], scores[:1]), batch(scores[1:], scores[1:])])
        tau, count, total, rate = result
        self.assertAlmostEqual(tau, .93, places=6)
        self.assertEqual((count, total), (10, 10))
        self.assertAlmostEqual(rate, .3, places=6)

    def test_minimum_and_agreement_across_classes(self):
        # Third sample disagrees. First two predict different classes.
        tau, count, total, _ = self.scan(
            Teacher(), [batch([.99, .20, .99], [.60, .10, .01])])
        self.assertAlmostEqual(tau, .80, places=6)
        self.assertEqual((count, total), (2, 3))

    def test_empty_disagreement_and_nonfinite_fallback(self):
        for loader in ([], [batch([.9], [.1])],
                       [(tensor([[np.nan, 0]]), tensor([[np.nan, 0]]))]):
            self.assertEqual(self.scan(Teacher(), loader)[0:2], (.85, 0))

    def test_no_floor_ceiling_or_cross_epoch_memory(self):
        for score in (.60, .99):
            tau, count, _, rate = self.scan(Teacher(), [batch([score], [score])])
            self.assertAlmostEqual(tau, score, places=6)
            self.assertEqual(count, 1)
            self.assertEqual(rate, 1.)

    def test_shared_threshold_inclusive_boundary_and_ties(self):
        logits1, logits2, _ = batch([.9, .1, .9], [.9, .1, .1])
        tau, _, _, rate = self.scan(Teacher(), [(logits1, logits2)])
        _, mask, agree = self.select(logits1, logits2, tensor([tau, tau]))
        self.assertEqual(mask.tolist(), [1., 1., 0.])
        self.assertEqual(agree, 2)
        self.assertEqual(rate, 1.)  # Ties can exceed the nominal top-30% fraction.

    def test_invalid_parameters(self):
        for q in (0., 1., -.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.scan(Teacher(), [], quantile=q)
        for fallback in (0., 1.1, np.nan):
            with self.assertRaises(ValueError):
                self.scan(Teacher(), [], fallback=fallback)


if __name__ == '__main__':
    unittest.main()
