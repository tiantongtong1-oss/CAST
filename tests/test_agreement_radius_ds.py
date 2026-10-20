"""Run the actual v18 batch gate with NumPy tensors and real source DS evidence.

No torch, GPU or dataset is needed. This checks decisions and diagnostic rows,
not just a second implementation of the acceptance expression.
"""
import argparse
import ast
import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from adaptive_confidence import DualViewClassMeans
from dual_domain_ds import DSConfig
from neighbor_support import MeanSupportBank as EvidenceBank, audit_mean_support


class Tensor(np.ndarray):
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def argmax(self, dim=None):
        return tensor(np.asarray(self).argmax(axis=dim))

    @property
    def device(self):
        return 'cpu'

    def to(self, **kwargs):
        return self

    def index_select(self, dim, indices):
        return tensor(np.take(self, indices, axis=dim))

    def bool(self):
        return tensor(self.astype(bool))

    def float(self):
        return tensor(self.astype(float))

    def max(self, dim=None, **kwargs):
        if dim is None:
            return np.asarray(self).max(**kwargs)
        return tensor(np.asarray(self).max(axis=dim)), tensor(np.asarray(self).argmax(axis=dim))

    def eq(self, other):
        return tensor(self == other)


def tensor(value):
    return np.asarray(value).view(Tensor)


class AgreementRadiusDSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path('train.py').read_text()
        cls.tree = ast.parse(cls.source)

    def parse(self, argv):
        nodes = [n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('parse_args', 'ds_config_from_args')]
        env = dict(argparse=argparse, np=np, DSConfig=DSConfig)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<args>', 'exec'), env)
        return env['parse_args'](argv)

    def run_gate(self, logits1, logits2, region, features, bank, enable_rescue=True, thresholds=(.8, .8)):
        # Extract the live selection and audit statements from the training body.
        start = self.source.index('                pseudo_targets, confidence_idx, agree_count =')
        end = self.source.index('                teacher_mean_features =', start)
        gate_start = self.source.index('                radius_candidates =')
        gate_end = self.source.index('                # Learning is distinct', gate_start)
        import textwrap
        code = textwrap.dedent(self.source[start:end]) + textwrap.dedent(self.source[gate_start:gate_end])
        def softmax(x, dim):
            z = np.exp(x - np.max(x, axis=dim, keepdims=True))
            return tensor(z / z.sum(axis=dim, keepdims=True))
        fake_torch = SimpleNamespace(
            zeros_like=lambda x: tensor(np.zeros_like(x)),
            where=lambda x: (tensor(np.where(x)[0]),),
            as_tensor=lambda x, **kw: tensor(x), bool=bool,
            minimum=lambda x, y: tensor(np.minimum(x, y)),
            max=lambda x, dim: (tensor(np.max(x, axis=dim)), tensor(np.argmax(x, axis=dim))))
        selector_tree = ast.parse(Path('ema_utils.py').read_text())
        selector = next(n for n in selector_tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'select_dual_view_pseudo_labels')
        selector.decorator_list = []
        selector_env = dict(torch=fake_torch, F=SimpleNamespace(softmax=softmax))
        exec(compile(ast.Module(body=[selector], type_ignores=[]), '<actual-selector>', 'exec'), selector_env)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audit.csv'
            env = dict(np=np, csv=csv, audit_mean_support=audit_mean_support, support_threshold=.6, i=0, device='cpu',
                       thresholds=tensor(thresholds), enable_radius_rescue=enable_rescue,
                       select_dual_view_pseudo_labels=selector_env['select_dual_view_pseudo_labels'],
                       logits1=tensor(logits1), logits2=tensor(logits2),
                       region_ok=tensor(region), teacher_mean_features=tensor(features),
                       sample_ids=tensor(np.arange(len(region))), ds_source=bank,
                       ds_config=DSConfig(k=3, min_neighbors=3, alpha=1.0),
                       ds_totals=np.zeros(7), ds_csv_path=str(path),
                       torch=fake_torch)
            exec(compile(code, '<actual-training-gate>', 'exec'), env)
            rows = list(csv.reader(io.StringIO(path.read_text()))) if path.exists() else []
        return env, rows

    def test_direct_confidence_and_ds_rescue(self):
        bank = EvidenceBank(np.tile([1., 0.], (3, 1)), [0, 0, 0],
                            [0, 1, 2], [1., 1., 1.], 2)
        # Low confidence rescued; high confidence bypasses radius; missing/opposite
        # DS evidence rejects low confidence; disagreement is always rejected.
        a = [[.01, 0], [10, 0], [.01, 0], [10, 0], [0, .01]]
        b = [[.02, 0], [10, 0], [.02, 0], [0, 10], [0, .02]]
        env, rows = self.run_gate(a, b, [True, False, True, True, True],
                                  [[1., 0.], [-1., 0.], [-1., 0.], [1., 0.], [1., 0.]], bank)
        self.assertEqual(env['final_mask'].tolist(), [True, True, False, False, False])
        self.assertEqual(env['ds_totals'][0], 3)
        self.assertEqual(env['ds_totals'][2], 1)
        self.assertEqual([row[1] for row in rows], ['0', '2', '4'])
        self.assertEqual([row[-1] for row in rows], ['True', 'False', 'False'])
        cols = next(n.value for n in ast.walk(self.tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'ds_columns' for t in n.targets))
        self.assertTrue(all(len(row) == len(cols.elts) for row in rows))

    def test_warmup_keeps_only_direct_confidence(self):
        class ForbiddenBank:
            def evaluate(self, *args):
                raise AssertionError('Warmup must skip DS')
        env, rows = self.run_gate([[.01, 0], [10, 0]], [[.01, 0], [10, 0]],
                                  [True, False], [[1., 0.], [1., 0.]], ForbiddenBank(), False)
        self.assertEqual(env['final_mask'].tolist(), [False, True])
        self.assertEqual(rows, [])

    def test_boundary_and_missing_class_close_direct_route(self):
        class ForbiddenBank:
            def evaluate(self, *args):
                raise AssertionError('Warmup must skip DS')
        env, _ = self.run_gate([[0, 0], [0, 10]], [[0, 0], [0, 10]],
                              [True, True], [[1., 0.], [1., 0.]], ForbiddenBank(), False,
                              thresholds=(.5, np.inf))
        self.assertEqual(env['final_mask'].tolist(), [True, False])

    def test_no_candidates_never_queries_ds(self):
        class ForbiddenBank:
            def evaluate(self, *args):
                raise AssertionError('No candidates should mean no DS query')
        env, rows = self.run_gate([[1, 0]], [[0, 1]], [True], [[1., 0.]], ForbiddenBank())
        self.assertFalse(env['final_mask'].any())
        self.assertEqual(rows, [])

    def test_default_is_ds_enabled_and_old_gates_rejected(self):
        self.assertTrue(self.parse([]).ds_neighbor)
        for argv in (['--phi', '1.4'],
                     ['--rescue_js_threshold', '.01'], ['--rescue_js_start_epoch', '20'],
                     ['--radius_rescue_warmup_epochs', '-1'], ['--no_radius_rescue']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(argv)

    def test_minimum_score_matches_two_inclusive_comparisons(self):
        rng = np.random.default_rng(17)
        conf1, conf2, threshold = rng.random((3, 500))
        conf1[:3] = [.8, .9, .7]
        conf2[:3] = [.9, .8, .8]
        threshold[:3] = .8
        np.testing.assert_array_equal(np.minimum(conf1, conf2) >= threshold,
                                      (conf1 >= threshold) & (conf2 >= threshold))

    def test_epoch_refresh_and_no_js_selection(self):
        self.assertNotIn('rescue_js', self.source)
        self.assertIn('final_mask = confidence_mask | rescue_mask', self.source)
        self.assertEqual(self.parse([]).radius_rescue_warmup_epochs, 3)
        args = self.parse([])
        self.assertEqual((args.threshold_base, args.threshold_beta, args.threshold_margin,
                          args.threshold_min, args.threshold_max), (.85, .5, .02, .8, .95))
        self.assertEqual(self.parse(['--threshold_base', '.9']).threshold_base, .9)
        loop = self.source.index('for i in range(args.epochs):')
        refresh = self.source.index('ds_source = refresh_ds_source_bank(', loop)
        batches = self.source.index('for weak1, weak2, strong, _, sample_ids', loop)
        self.assertLess(refresh, batches)
        self.assertLess(self.source.index('thresholds, class_mean, global_mean, threshold_center = calculate_target_thresholds(', loop), batches)
        scanner = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'calculate_target_thresholds')
        self.assertFalse(any(isinstance(n, ast.Name) and n.id == 'unused_target_labels'
                             and isinstance(n.ctx, ast.Load) for n in ast.walk(scanner)))
        self.assertIn('if i % args.source_refresh_interval == 0:', self.source)
        self.assertIn('prototype_update_mask |= final_mask', self.source)
        self.assertIn('criterion(output[0], train_targets) * train_con_idx', self.source)


class ThresholdStatisticsTest(unittest.TestCase):
    def test_unequal_batches_agreement_and_missing_class(self):
        stats = DualViewClassMeans(3)
        stats.update([[.8,.1,.1]], [[.7,.2,.1]])
        stats.update([[.9,.05,.05], [.6,.2,.2], [.1,.8,.1], [.8,.1,.1]],
                     [[.8,.1,.1], [.5,.3,.2], [.2,.7,.1], [.1,.8,.1]])
        thresholds, counts = stats.finalize()
        np.testing.assert_allclose(thresholds[:2], [2./3., .7])
        self.assertTrue(np.isinf(thresholds[2]))
        self.assertEqual(counts.tolist(), [3, 1, 0])

    def test_empty_epoch_and_no_statistics_leak_across_epochs(self):
        stats = DualViewClassMeans(2)
        stats.update(np.empty((0,2)), np.empty((0,2)))
        stats.update([[.9,.1]], [[.1,.9]])
        self.assertTrue(np.isinf(stats.finalize()[0]).all())
        stats.update([[.9,.1]], [[.8,.2]])
        self.assertAlmostEqual(stats.finalize()[0][0], .8)
        self.assertTrue(np.isinf(DualViewClassMeans(2).finalize()[0]).all())

if __name__ == '__main__':
    unittest.main()
