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

    def std(self, unbiased=False):
        return np.asarray(self).std(ddof=1 if unbiased else 0)

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


class MeanStdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = staticmethod(load_function('train.py', 'calculate_global_mean_std_threshold'))

    def test_pooled_mean_std_and_rate(self):
        scores = [.60, .70, .80, .90, .99]
        r = self.scan(Teacher(), [batch(scores[:1], scores[:1]), batch(scores[1:], scores[1:])])
        expected = np.mean(scores) + .5 * np.std(scores)
        self.assertAlmostEqual(r[0], expected, places=6)
        self.assertEqual(r[1:3], (5, 5))
        self.assertAlmostEqual(r[3], 2/5, places=6)

    def test_smoothing_across_epochs(self):
        first = self.scan(Teacher(), [batch([.9], [.9])])[0]
        second = self.scan(Teacher(), [batch([.99], [.99])], previous_tau=first)[0]
        self.assertAlmostEqual(first, .9, places=6)
        self.assertAlmostEqual(second, .918, places=6)

    def test_bounds_and_single_sample(self):
        for score, expected in [(.6, .85), (.999, .99)]:
            r = self.scan(Teacher(), [batch([score], [score])])
            self.assertAlmostEqual(r[0], expected, places=6)
            self.assertEqual(r[5], 0.)

    def test_empty_hold_and_initial_fallback(self):
        for loader in ([], [batch([.9], [.1])],
                       [(tensor([[np.nan, 0]]), tensor([[np.nan, 0]]))]):
            self.assertEqual(self.scan(Teacher(), loader)[:2], (.85, 0))
            self.assertEqual(self.scan(Teacher(), loader, previous_tau=.92)[:2], (.92, 0))

    def test_agreement_minimum_and_label_independence(self):
        r = self.scan(Teacher(), [batch([.99,.20,.99], [.60,.10,.01])],
                      threshold_min=.5, fallback=.5)
        self.assertEqual(r[1:3], (2, 3))
        self.assertAlmostEqual(r[4], .7, places=6)
        self.assertAlmostEqual(r[0], .75, places=6)

    def test_invalid_settings(self):
        for kw in ({'std_scale': -1}, {'std_scale': np.nan},
                   {'smoothing': 1}, {'smoothing': -.1},
                   {'threshold_min': .9}, {'threshold_max': np.nan},
                   {'previous_tau': .1}, {'previous_tau': np.nan}):
            with self.assertRaises(ValueError):
                self.scan(Teacher(), [], **kw)


if __name__ == '__main__':
    unittest.main()
