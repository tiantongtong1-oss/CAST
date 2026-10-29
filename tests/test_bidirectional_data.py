"""Exercise actual dataset bodies/routing without requiring PyTorch or OpenCV."""
import argparse
import ast
import contextlib
import glob
import io
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


def load_datasets():
    tree = ast.parse(Path('dataset.py').read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    env = dict(np=np, pd=pd, os=os, glob=glob, random=random,
               data=SimpleNamespace(Dataset=object),
               cv2=SimpleNamespace(imread=lambda path: np.zeros((4, 4, 3))),
               util=SimpleNamespace(**{n: lambda x: x for n in
                                      ('flip_image', 'add_gaussian_noise', 'crop', 'rotation')}))
    exec(compile(tree, 'dataset.py', 'exec'), env)
    return env


class BidirectionalTests(unittest.TestCase):
    def setUp(self):
        self.env = load_datasets()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'EmoLabel').mkdir()
        rows = []
        for c in range(7):
            rows.extend(f'train_{c}_{i}.jpg {c+1}\n' for i in range(10))
            rows.extend(f'test_{c}_{i}.jpg {c+1}\n' for i in range(2))
        (self.root / 'EmoLabel/list_patition_label.txt').write_text(''.join(rows))

    def raf(self, phase, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.env['RafDataSet'](str(self.root), phase, **kwargs)

    def test_holdout_disjoint_complete_and_reproducible_ids(self):
        train = self.raf('train', val_fraction=.1)
        scan = self.raf('train', val_fraction=.1)
        val = self.raf('val', val_fraction=.1)
        test = self.raf('test', val_fraction=.1)
        self.assertEqual((len(train), len(val), len(test)), (63, 7, 14))
        self.assertEqual(train.file_paths, scan.file_paths)
        self.assertFalse(set(train.file_paths) & set(val.file_paths))
        self.assertFalse((set(train.file_paths) | set(val.file_paths)) & set(test.file_paths))
        self.assertEqual(set(train.file_paths) | set(val.file_paths),
                         set(self.raf('train').file_paths))
        self.assertEqual(val.label_dis, [1]*7)
        for name, label in zip(train.file_paths, train.label):
            self.assertEqual(int(Path(name).name.split('_')[1]), label)
        before = np.random.get_state()
        self.raf('train', val_fraction=.1)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])

    def test_raf_target_return_contracts_independent_weak_calls(self):
        calls = []
        def weak(image):
            calls.append(1)
            return len(calls)
        ds = self.raf('train', transform=weak, weak2_transform=weak,
                      strong_transform=lambda image: 'strong', return_index=True,
                      val_fraction=.1)
        item = ds[0]
        self.assertEqual(item[:3], (1, 2, 'strong'))
        self.assertEqual(item[-1], 0)
        self.assertEqual(len(item), 5)
        self.assertEqual(len(self.raf('train', transform=weak, weak2_transform=weak)[0]), 3)
        self.assertEqual(len(self.raf('train', transform=weak)[0]), 2)

    def test_fer_source_mapping(self):
        for c in range(7):
            folder = self.root / 'train' / str(c)
            folder.mkdir(parents=True)
            (folder / 'a.jpg').touch()
        with contextlib.redirect_stdout(io.StringIO()):
            ds = self.env['FER'](str(self.root), 'train', transform=lambda x: x)
        self.assertEqual(len(ds[0]), 2)
        for path, label in zip(ds.file_paths, ds.label):
            self.assertEqual(label, ds.FER_TO_CAST[int(Path(path).parent.name)])

    def test_both_training_routes(self):
        tree = ast.parse(Path('train.py').read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_training')
        start = next(i for i,n in enumerate(run.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'dataset_classes' for t in n.targets))
        end = next(i for i,n in enumerate(run.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'class_num' for t in n.targets))
        class Dataset:
            def __init__(self, path, **kw): self.path, self.kw = path, kw
            def __len__(self): return 10
        class Raf(Dataset): pass
        class Fer(Dataset): pass
        for source, target in [('rafdb', 'fer'), ('fer', 'rafdb')]:
            env = dict(args=SimpleNamespace(data1=source, data2=target,
                       source_path='source', target_path='target',
                       raf_val_fraction=.1, raf_split_seed=2000),
                       RafDataSet=Raf, FER=Fer, weak_transform='weak',
                       strong_transform='strong', test_transform='test')
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(ast.Module(body=run.body[start:end], type_ignores=[]),
                             'train.py', 'exec'), env)
            self.assertIsInstance(env['source_train'], Raf if source == 'rafdb' else Fer)
            self.assertIsInstance(env['target_train'], Raf if target == 'rafdb' else Fer)
            self.assertTrue(env['target_train'].kw['return_index'])
            self.assertEqual(env['target_threshold'].kw['weak2_transform'], 'weak')
            self.assertEqual(env['target_val'].kw['phase'], 'val')
            self.assertEqual(env['target_test'].kw['phase'], 'test')
            self.assertEqual('val_fraction' in env['target_train'].kw, target == 'rafdb')

    def test_parser_and_pretraining_regression(self):
        tree = ast.parse(Path('train.py').read_text())
        parse = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
        env = dict(argparse=argparse, np=np, ds_config_from_args=lambda args: None)
        exec(compile(ast.Module(body=[parse], type_ignores=[]), 'train.py', 'exec'), env)
        args = env['parse_args'](['--data1','fer','--data2','rafdb','--pre_epochs','30'])
        self.assertEqual(args.raf_val_fraction, .1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            env['parse_args'](['--data1','fer','--data2','fer'])
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_training')
        loop = next(n for n in run.body if isinstance(n, ast.For)
                    and 'args.pre_epochs' in ast.unparse(n.iter))
        self.assertNotIn('target_weight', ast.unparse(loop))


if __name__ == '__main__':
    unittest.main()
