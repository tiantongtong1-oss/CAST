"""Numerical and train-wiring tests runnable without torch/datasets."""
import ast
import itertools
from pathlib import Path
import unittest
import numpy as np
from dual_domain_ds import DSConfig, EvidenceBank, combine_domains, fuse_simple_support


class DSTest(unittest.TestCase):
    def bank(self, labels, quality=None, ids=None):
        n = len(labels)
        return EvidenceBank(np.tile([1., 0.], (n, 1)), labels,
                            np.arange(n) if ids is None else ids,
                            np.ones(n) if quality is None else quality, 2)

    def test_vacuous_and_identical(self):
        b, u, k = fuse_simple_support([], [], 2)
        np.testing.assert_allclose(b, 0)
        self.assertEqual((u, k), (1, 0))
        b, u, k = fuse_simple_support([0, 0], [.6, .6], 2)
        np.testing.assert_allclose(b, [.84, 0])
        self.assertAlmostEqual(u, .16)
        self.assertAlmostEqual(k, 0)

    def test_conflict_hand_calculation(self):
        b, u, k = fuse_simple_support([0, 1], [.8, .7], 2)
        np.testing.assert_allclose(b, [.24/.44, .14/.44])
        self.assertAlmostEqual(u, .06/.44)
        self.assertAlmostEqual(k, .56)

    def test_order_and_bruteforce(self):
        labels, r = [0, 1, 0], [.7, .2, .5]
        b, u, k = fuse_simple_support(labels, r, 2)
        raw = np.zeros(3); conflict = 0.
        for bits in itertools.product([False, True], repeat=3):
            probability = np.prod([v if bit else 1-v for v, bit in zip(r, bits)])
            chosen = {labels[i] for i, bit in enumerate(bits) if bit}
            if len(chosen) > 1:
                conflict += probability
            else:
                raw[next(iter(chosen)) if chosen else 2] += probability
        np.testing.assert_allclose(np.r_[b, u], raw/raw.sum())
        self.assertAlmostEqual(k, conflict)
        for perm in itertools.permutations(range(3)):
            bb, uu, kk = fuse_simple_support(np.array(labels)[list(perm)], np.array(r)[list(perm)], 2)
            np.testing.assert_allclose(bb, b)
            self.assertAlmostEqual(uu, u); self.assertAlmostEqual(kk, k)

    def test_total_conflict_is_not_nan(self):
        b, u, k = fuse_simple_support([0, 1], [1, 1], 2)
        self.assertEqual((u, k), (1, 1))
        np.testing.assert_allclose(b, 0)

    def test_missing_self_and_invalid_query(self):
        cfg = DSConfig(k=3, min_neighbors=3)
        bank = self.bank([0, 0, 0])
        r = bank.evaluate([[1., 0.]], [0], cfg, exclude_self=True)
        self.assertEqual(r['count'][0], 2)
        self.assertFalse(r['available'][0])
        r = bank.evaluate([[0., 0.], [np.nan, 1.]], [99, 100], cfg)
        self.assertFalse(r['available'].any())
        empty = self.bank([]).evaluate([[1., 0.]], [99], cfg)
        self.assertEqual(empty['uncertainty'][0], 1)

    def test_cross_domain_ids_are_not_self_ids(self):
        cfg = DSConfig(k=3, min_neighbors=3)
        r = self.bank([0, 0, 0]).evaluate([[1., 0.]], [0], cfg)
        self.assertTrue(r['available'][0])

    def test_radius_and_duplicate_ids(self):
        cfg = DSConfig(k=3, min_neighbors=3)
        r = self.bank([0, 0, 0]).evaluate([[-1., 0.]], [99], cfg)
        self.assertEqual(r['count'][0], 0)
        with self.assertRaises(ValueError):
            self.bank([0, 0], ids=[2, 2])

    def test_agreement_passes_opposition_and_missing_fail(self):
        cfg = DSConfig(k=3, min_neighbors=3)
        a = self.bank([0, 0, 0]).evaluate([[1., 0.]], [99], cfg)
        b = self.bank([1, 1, 1]).evaluate([[1., 0.]], [99], cfg)
        self.assertTrue(combine_domains(a, a, [0], cfg)['pass_mask'][0])
        x = combine_domains(a, b, [0], cfg)
        self.assertFalse(x['pass_mask'][0]); self.assertGreater(x['cross_conflict'][0], .9)
        missing = self.bank([]).evaluate([[1., 0.]], [99], cfg)
        x = combine_domains(a, missing, [0], cfg)
        self.assertFalse(x['pass_mask'][0]); self.assertGreater(x['uncertainty'][0], .5)

    def test_within_domain_conflict_discounted_to_unknown(self):
        cfg = DSConfig(k=4, min_neighbors=3)
        r = self.bank([0, 0, 1, 1]).evaluate([[1., 0.]], [99], cfg)
        self.assertFalse(r['available'][0]); self.assertGreater(r['conflict'][0], .5)
        self.assertEqual(r['uncertainty'][0], 1)

    def test_low_quality_increases_unknown(self):
        cfg = DSConfig(k=3, min_neighbors=3)
        r = self.bank([0]*3, [.01]*3).evaluate([[1., 0.]], [99], cfg)
        self.assertTrue(r['available'][0])
        self.assertFalse(combine_domains(r, r, [0], cfg)['pass_mask'][0])

    def test_finite_normalized_mass(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            b, u, k = fuse_simple_support(rng.integers(0, 7, 20), rng.uniform(0, .95, 20), 7)
            self.assertTrue(np.isfinite(b).all())
            self.assertAlmostEqual(b.sum()+u, 1)
            self.assertTrue(0 <= k <= 1)

    def test_invalid_config(self):
        for kwargs in ({'k': 0}, {'alpha': float('nan')}, {'bandwidth': 0}, {'strength_cap': 1}):
            with self.assertRaises(ValueError):
                DSConfig(**kwargs)

    def test_train_argument_validation_without_dependencies(self):
        import argparse
        tree = ast.parse(Path('train.py').read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in ('parse_args', 'ds_config_from_args')]
        namespace = dict(argparse=argparse, np=np, DSConfig=DSConfig)
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<args>', 'exec'), namespace)
        args = namespace['parse_args'](['--ds_neighbor'])
        self.assertTrue(args.ds_neighbor)
        self.assertTrue(namespace['parse_args'](['--ds_neighbor', '--no_role_separation']).ds_neighbor)
        with self.assertRaises(SystemExit):
            namespace['parse_args'](['--ds_source_weight', '0.5'])

    def test_wiring_scope_and_no_truth(self):
        s = Path('train.py').read_text()
        self.assertLess(s.index('radius_candidates = agreement_mask & region_ok'), s.index('source_ds = ds_source.evaluate'))
        self.assertLess(s.index('source_ds = ds_source.evaluate'), s.index('final_mask[query_idx] = torch.as_tensor'))
        self.assertEqual(s.count('n = max(ds_totals[0], 1)'), 1)
        self.assertGreater(s.index('n = max(ds_totals[0], 1)'), s.index('for i in range(args.epochs):'))
        tree = ast.parse(s)
        refresh = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'refresh_ds_source_bank')
        reads = [n for n in ast.walk(refresh) if isinstance(n, ast.Name)
                 and n.id == '_ignored_target_truth' and isinstance(n.ctx, ast.Load)]
        self.assertFalse(reads)


if __name__ == '__main__':
    unittest.main()

