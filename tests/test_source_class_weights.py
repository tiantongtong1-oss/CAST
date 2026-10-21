import ast
from pathlib import Path
import unittest
import numpy as np
from source_class_weights import source_class_weights


class SourceWeightTests(unittest.TestCase):
    def test_inverse_sqrt_ratio_and_normalization(self):
        counts = np.array([1290, 281, 717, 4772, 1982, 705, 2524])
        weights = source_class_weights(counts, 'sqrt_inverse')
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertAlmostEqual(weights[1] / weights[3], np.sqrt(4772 / 281))
        self.assertTrue(np.isfinite(weights).all())

    def test_none_and_equal_counts(self):
        np.testing.assert_array_equal(source_class_weights([1, 100]), [1, 1])
        np.testing.assert_array_equal(source_class_weights([5, 5], 'sqrt_inverse'), [1, 1])

    def test_invalid_counts(self):
        for counts in ([], [0, 2], [-1, 2], [np.nan, 2], [[1, 2]]):
            with self.assertRaises(ValueError):
                source_class_weights(counts, 'sqrt_inverse')

    def test_source_loop_has_no_target_statistics(self):
        tree = ast.parse(Path('train.py').read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                    and isinstance(n.iter, ast.Call)
                    and ast.unparse(n.iter) == 'range(args.pre_epochs)')
        names = {n.id for n in ast.walk(loop) if isinstance(n, ast.Name)}
        self.assertNotIn('rescue_accept_num', names)
        self.assertNotIn('ds_totals', names)
        self.assertIn('source_criterion', names)

    def test_joint_loss_uses_source_weights_only(self):
        # Execute the real loss statements with a NumPy-backed tensor stand-in.
        class Tensor(np.ndarray):
            def clamp_min(self, minimum):
                return np.maximum(self, minimum)
        def tensor(x):
            return np.asarray(x).view(Tensor)
        tree = ast.parse(Path('train.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_training')
        loop = next(n for n in ast.walk(function) if isinstance(n, ast.For)
                    and ast.unparse(n.target) == '(weak1, weak2, strong, _, sample_ids)')
        start = next(i for i,n in enumerate(loop.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == 'source_count')
        end = next(i for i,n in enumerate(loop.body[start:], start) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == 'cls_loss')
        code = compile(ast.Module(body=loop.body[start:end+1], type_ignores=[]), '<actual-loss>', 'exec')
        for weights in ([1., 1.], [2., .5]):
            env = dict(source_imgs=tensor([[0], [0]]), train_targets=tensor([0,1,1,0]),
                       train_con_idx=tensor([1.,1.,1.,0.]), output=[tensor([2.,4.,6.,99.])],
                       source_weights=tensor(weights), criterion=lambda x,y:x,
                       source_criterion=lambda x,y:x*tensor(weights)[y])
            exec(code, env)
            self.assertAlmostEqual(env['cls_loss'], (2*weights[0]+4*weights[1]+6)/(sum(weights)+1))
