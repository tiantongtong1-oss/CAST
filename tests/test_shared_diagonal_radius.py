import argparse
import ast
import contextlib
import io
from pathlib import Path
import unittest
import numpy as np
from shared_diagonal_radius import fit_shared_diagonal
from balanced_radius import balanced_radius_quantile
from dual_domain_ds import DSConfig


class SharedDiagonalTests(unittest.TestCase):
    def test_class_balance_and_within_class_variance(self):
        # Unit vectors; class 0 varies along x, class 1 is a constant along y.
        a = np.array([[1., 0.], [-1., 0.]])
        b = np.array([[0., 1.]])
        means, variance, radius, distances = fit_shared_diagonal([a, b], .9, .1)
        np.testing.assert_allclose(means, [[0., 0.], [0., 1.]])
        # Only within-class residuals: no variance due to different class means.
        np.testing.assert_allclose(variance, [.5, 0.])
        np.testing.assert_allclose(distances[0], [np.sqrt(1/.6)] * 2)
        np.testing.assert_allclose(distances[1], [0.])
        self.assertAlmostEqual(radius, np.sqrt(1/.6))
        replicated = fit_shared_diagonal([np.tile(a, (100, 1)), b], .9, .1)
        np.testing.assert_allclose(replicated[0], means)
        np.testing.assert_allclose(replicated[1], variance)
        self.assertAlmostEqual(replicated[2], radius)

    def test_weighted_threshold_and_regularization(self):
        rng = np.random.default_rng(19)
        classes = [rng.normal(size=(n, 4)) for n in (10, 90, 3)]
        classes = [x / np.linalg.norm(x, axis=1, keepdims=True) for x in classes]
        means, variance, radius, distances = fit_shared_diagonal(classes, .7, 1e-4)
        expected_var = np.mean([np.var(x, axis=0) for x in classes], axis=0)
        np.testing.assert_allclose(variance, expected_var)
        for x, mu, d in zip(classes, means, distances):
            np.testing.assert_allclose(d, np.sqrt(((x-mu)**2/(expected_var+1e-4)).sum(1)))
        self.assertEqual(radius, balanced_radius_quantile(distances, .7))

    def test_empty_missing_and_zero_variance(self):
        empty = np.empty((0, 2))
        means, var, radius, distances = fit_shared_diagonal([empty, [[1., 0.]]])
        np.testing.assert_array_equal(var, [0., 0.])
        self.assertEqual(radius, 0.)
        self.assertEqual(len(distances[0]), 0)
        means, var, radius, _ = fit_shared_diagonal([empty, empty])
        self.assertEqual(radius, 0.)
        self.assertTrue(np.isfinite(means).all())

    def test_invalid_inputs(self):
        for value in (0, -1, True, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fit_shared_diagonal([[[1., 0.]]], regularizer=value)
        for q in (0, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                fit_shared_diagonal([[[1., 0.]]], quantile=q)
        for arrays in ([], [np.array([1., 0.])], [np.zeros((1, 2)), np.zeros((1, 3))],
                       [np.array([[float('nan'), 0.]])]):
            with self.assertRaises(ValueError):
                fit_shared_diagonal(arrays)

    def test_live_arguments(self):
        tree = ast.parse(Path('train.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('parse_args', 'ds_config_from_args')]
        env = dict(argparse=argparse, np=np, DSConfig=DSConfig)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<args>', 'exec'), env)
        parse = env['parse_args']
        args = parse([])
        self.assertEqual(args.source_radius_quantile, .9)
        self.assertEqual(args.source_variance_regularizer, 1e-4)
        self.assertEqual(parse(['--source_variance_regularizer', '.001']).source_variance_regularizer, .001)
        for argv in (['--source_variance_regularizer', '0'], ['--source_variance_regularizer', 'nan'],
                     ['--source_radius_quantile', '0'], ['--source_radius_std_scale', '1']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse(argv)


if __name__ == '__main__':
    unittest.main()
