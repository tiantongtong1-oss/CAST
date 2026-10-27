"""Check the live classification block against original CAST's masked batch mean."""
import ast
from pathlib import Path
import textwrap
import unittest
from types import SimpleNamespace
import numpy as np


class Tensor(np.ndarray):
    @property
    def device(self):
        return 'cpu'

    def to(self, **kwargs):
        return np.asarray(self, dtype=kwargs['dtype']).view(Tensor)


def tensor(x):
    return np.asarray(x, dtype=float).view(Tensor)


class CastClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path('train.py').read_text()
        start = source.index('            per_sample_loss = criterion(output[0], train_targets)',
                             source.index('# CAST-equivalent CE'))
        end = source.index('            aff_loss = output[1]', start)
        cls.code = compile(textwrap.dedent(source[start:end]), '<actual-CE>', 'exec')
        cls.source = source

    def run_loss(self, ns, losses, mask):
        env = dict(criterion=lambda logits, labels: tensor(losses),
                   output=[None], train_targets=None,
                   source_imgs=np.empty((ns, 1)), con_idx=tensor(mask))
        exec(self.code, env)
        expected = np.mean(np.asarray(losses) * np.r_[np.ones(ns), mask])
        self.assertAlmostEqual(float(env['cls_loss']), expected, places=12)
        return env

    def test_unequal_batches_and_partial_acceptance(self):
        env = self.run_loss(2, [1, 3, 2, 999, 4], [1, 0, 1])
        self.assertAlmostEqual(float(env['source_cls_loss']), 2)
        self.assertAlmostEqual(float(env['target_cls_loss']), 2)
        self.assertAlmostEqual(float(env['cls_loss']), 2)

    def test_no_target_accepted_preserves_all_sample_denominator(self):
        env = self.run_loss(2, [1, 3, 999, 999, 999], [0, 0, 0])
        self.assertAlmostEqual(float(env['cls_loss']), .8)

    def test_all_targets_accepted(self):
        self.run_loss(3, [1, 2, 3, 4, 5], [1, 1])

    def test_equal_domain_batches(self):
        env = self.run_loss(2, [1, 3, 2, 4], [1, 0])
        self.assertAlmostEqual(float(env['cls_loss']),
                               .5 * env['source_cls_loss'] + .5 * env['target_cls_loss'])

    def test_many_batch_sizes_and_masks(self):
        rng = np.random.default_rng(8)
        for ns in (1, 3, 32):
            for nt in (1, 7, 32):
                for _ in range(10):
                    self.run_loss(ns, rng.random(ns + nt), rng.integers(0, 2, nt))

    def test_removed_ramp_and_unreduced_criterion(self):
        self.assertNotIn('target_weight', self.source)
        self.assertIn("CrossEntropyLoss(reduction='none')", self.source)


if __name__ == '__main__':
    unittest.main()
