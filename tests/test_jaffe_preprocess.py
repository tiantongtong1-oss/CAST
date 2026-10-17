import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import argparse
import numpy as np
from prepare_jaffe import TEMPLATE, similarity_transform, choose_face
from training_log import training_log


class AlignmentTests(unittest.TestCase):
    def test_identity_and_known_scale_rotation(self):
        np.testing.assert_allclose(similarity_transform(TEMPLATE, TEMPLATE),
                                   [[1, 0, 0], [0, 1, 0]], atol=1e-10)
        a = .2
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        target = 2 * TEMPLATE @ rotation.T + [4, -3]
        matrix = similarity_transform(TEMPLATE, target)
        np.testing.assert_allclose(TEMPLATE @ matrix[:, :2].T + matrix[:, 2], target, atol=1e-10)
        self.assertGreater(np.linalg.det(matrix[:, :2]), 0)

    def test_bad_landmarks_and_no_detection(self):
        with self.assertRaises(ValueError): similarity_transform(np.zeros((5, 2)), TEMPLATE)
        with self.assertRaises(ValueError): choose_face({})
        small = dict(facial_area=[0, 0, 10, 10], score=.99)
        large = dict(facial_area=[0, 0, 20, 20], score=.95)
        self.assertIs(choose_face({'a': small, 'b': large}), large)

    def test_jaffe_cli_allows_source_training_without_checkpoint(self):
        from dual_domain_ds import DSConfig
        tree = ast.parse(Path('train.py').read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {'parse_args', 'ds_config_from_args'}]
        env = dict(argparse=argparse, np=np, DSConfig=DSConfig)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), 'train.py', 'exec'), env)
        args = env['parse_args'](['--data2', 'jaffe', '--pre_epochs', '30', '--epochs', '30'])
        self.assertIsNone(args.checkpoint)
        self.assertEqual(args.pre_epochs, 30)

    def test_log_captures_output_without_overwriting(self):
        with TemporaryDirectory() as directory:
            with training_log(directory, 'test'):
                print('training line')
            self.assertIn('training line', Path(directory, 'test.log').read_text())
            with self.assertRaises(FileExistsError):
                with training_log(directory, 'test'): pass


if __name__ == '__main__':
    unittest.main()
