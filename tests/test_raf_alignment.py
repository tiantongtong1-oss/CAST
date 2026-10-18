import ast
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from raf_alignment import (read_raf_labels, choose_input_directory, find_input,
                           aligned_raf_path, validate_shared_alignment)
from prepare_jaffe import TEMPLATE


class RafAlignmentTests(unittest.TestCase):
    def test_input_preference_and_label_integrity(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / 'EmoLabel').mkdir()
            (root / 'EmoLabel/list_patition_label.txt').write_text('train_00001.jpg 1\ntest_0001.jpg 6\n')
            (root / 'Image/aligned').mkdir(parents=True)
            (root / 'Image/aligned/train_00001_aligned.jpg').touch()
            self.assertEqual(choose_input_directory(root, 'auto')[0], 'aligned')
            (root / 'Image/original').mkdir()
            (root / 'Image/original/train_00001.jpg').touch()
            mode, folder = choose_input_directory(root, 'auto')
            self.assertEqual(mode, 'original')
            self.assertEqual(find_input(folder, 'train_00001.jpg', mode).name, 'train_00001.jpg')
            self.assertEqual(read_raf_labels(root), [('train_00001.jpg', 1), ('test_0001.jpg', 6)])

    def test_shared_metadata_mismatch_or_incomplete_rejected(self):
        meta = dict(method='RetinaFace five-point similarity', image_size=256,
                    template_112=TEMPLATE.tolist(), threshold=.9,
                    input_count=2, aligned_count=2, failed_count=0)
        with TemporaryDirectory() as d:
            roots = [Path(d)/'a', Path(d)/'b']
            for root in roots:
                root.mkdir()
                (root/'alignment_metadata.json').write_text(json.dumps(meta))
            validate_shared_alignment(*roots)
            bad = dict(meta, threshold=.8)
            (roots[1]/'alignment_metadata.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError): validate_shared_alignment(*roots)
            bad = dict(meta, failed_count=1, aligned_count=1)
            (roots[1]/'alignment_metadata.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError): validate_shared_alignment(*roots)

    def test_actual_raf_loader_keeps_labels_and_paths(self):
        tree = ast.parse(Path('dataset.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RafDataSet')
        env = dict(data=SimpleNamespace(Dataset=object), np=np, pd=pd, os=os,
                   aligned_raf_path=aligned_raf_path,
                   util=SimpleNamespace(flip_image=None, add_gaussian_noise=None, crop=None, rotation=None))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'dataset.py', 'exec'), env)
        with TemporaryDirectory() as d:
            root = Path(d)
            (root/'EmoLabel').mkdir()
            (root/'Image/aligned').mkdir(parents=True)
            (root/'EmoLabel/list_patition_label.txt').write_text('train_00001.jpg 2\ntest_0001.jpg 6\n')
            for name in ['train_00001.jpg', 'test_0001.jpg']:
                aligned_raf_path(root, name).touch()
            (root/'alignment_metadata.json').write_text(json.dumps(dict(
                input_count=2, aligned_count=2, failed_count=0, output_format='raf_aligned_png')))
            train = env['RafDataSet'](d, 'train')
            test = env['RafDataSet'](d, 'test')
            self.assertEqual(train.label.tolist(), [1])
            self.assertEqual(test.label.tolist(), [5])
            self.assertTrue(train.file_paths[0].endswith('train_00001_aligned.png'))


if __name__ == '__main__':
    unittest.main()
