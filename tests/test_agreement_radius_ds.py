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
from dual_domain_ds import DSConfig, EvidenceBank, audit_source


class Tensor(np.ndarray):
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def argmax(self, dim=None):
        return tensor(np.asarray(self).argmax(axis=dim))

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

    def run_gate(self, logits1, logits2, region, features, bank):
        # Extract the live selection and audit statements from the training body.
        start = self.source.index('                pseudo_targets = logits1.argmax')
        end = self.source.index('                teacher_mean_features =', start)
        gate_start = self.source.index('                radius_candidates =')
        gate_end = self.source.index('                # Learning is distinct', gate_start)
        import textwrap
        code = textwrap.dedent(self.source[start:end]) + textwrap.dedent(self.source[gate_start:gate_end])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audit.csv'
            env = dict(np=np, csv=csv, audit_source=audit_source, i=0, device='cpu',
                       logits1=tensor(logits1), logits2=tensor(logits2),
                       region_ok=tensor(region), teacher_mean_features=tensor(features),
                       sample_ids=tensor(np.arange(len(region))), ds_source=bank,
                       ds_config=DSConfig(k=3, min_neighbors=3, alpha=1.0),
                       ds_totals=np.zeros(7), ds_csv_path=str(path),
                       torch=SimpleNamespace(zeros_like=lambda x: tensor(np.zeros_like(x)),
                                             where=lambda x: (tensor(np.where(x)[0]),),
                                             as_tensor=lambda x, **kw: tensor(x), bool=bool))
            exec(compile(code, '<actual-training-gate>', 'exec'), env)
            rows = list(csv.reader(io.StringIO(path.read_text()))) if path.exists() else []
        return env, rows

    def test_all_gates_and_no_confidence_bypass(self):
        bank = EvidenceBank(np.tile([1., 0.], (3, 1)), [0, 0, 0],
                            [0, 1, 2], [1., 1., 1.], 2)
        # Low confidence passes; high confidence fails radius/DS; disagreement fails.
        a = [[.01, 0], [10, 0], [10, 0], [10, 0], [0, 10]]
        b = [[.02, 0], [10, 0], [10, 0], [0, 10], [0, 10]]
        env, rows = self.run_gate(a, b, [True, False, True, True, True],
                                  [[1., 0.], [1., 0.], [-1., 0.], [1., 0.], [1., 0.]], bank)
        self.assertEqual(env['final_mask'].tolist(), [True, False, False, False, False])
        self.assertEqual(env['ds_totals'][0], 3)
        self.assertEqual(env['ds_totals'][2], 1)
        self.assertEqual([row[1] for row in rows], ['0', '2', '4'])
        self.assertEqual([row[-1] for row in rows], ['True', 'False', 'False'])
        cols = next(n.value for n in ast.walk(self.tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'ds_columns' for t in n.targets))
        self.assertTrue(all(len(row) == len(cols.elts) for row in rows))

    def test_no_candidates_never_queries_ds(self):
        class ForbiddenBank:
            def evaluate(self, *args):
                raise AssertionError('No candidates should mean no DS query')
        env, rows = self.run_gate([[1, 0]], [[0, 1]], [True], [[1., 0.]], ForbiddenBank())
        self.assertFalse(env['final_mask'].any())
        self.assertEqual(rows, [])

    def test_default_is_ds_enabled_and_old_gates_rejected(self):
        self.assertTrue(self.parse([]).ds_neighbor)
        for argv in (['--threshold_base', '.9'], ['--phi', '1.4'],
                     ['--rescue_js_threshold', '.01'], ['--rescue_js_start_epoch', '20'],
                     ['--radius_rescue_warmup_epochs', '3'], ['--no_radius_rescue']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(argv)

    def test_epoch_zero_refresh_and_no_confidence_or_js_selection(self):
        self.assertNotIn('rescue_js', self.source)
        self.assertNotIn('calculate_target_thresholds', self.source)
        self.assertNotIn('confidence_mask', self.source)
        loop = self.source.index('for i in range(args.epochs):')
        refresh = self.source.index('ds_source = refresh_ds_source_bank(', loop)
        batches = self.source.index('for weak1, weak2, strong, _, sample_ids', loop)
        self.assertLess(refresh, batches)
        self.assertIn('if i % args.source_refresh_interval == 0:', self.source)
        self.assertIn('prototype_update_mask |= final_mask', self.source)
        self.assertIn('criterion(output[0], train_targets) * train_con_idx', self.source)


if __name__ == '__main__':
    unittest.main()
