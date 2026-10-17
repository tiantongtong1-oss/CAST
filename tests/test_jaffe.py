import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from jaffe_manifest import scan_jaffe, POSE_TO_CAST


class JaffeTests(unittest.TestCase):
    def test_original_filenames_and_all_pose_labels(self):
        with TemporaryDirectory() as root:
            for i, code in enumerate(POSE_TO_CAST):
                Path(root, 'KA.%s1.%d.tiff' % (code, i)).touch()
            records = scan_jaffe(root)
            self.assertEqual(sorted(y for _, y in records), list(range(7)))
            self.assertEqual(records, scan_jaffe(root))

    def test_bad_labels_and_copied_splits_rejected(self):
        with TemporaryDirectory() as root:
            p = Path(root, 'KA.XX1.tiff'); p.touch()
            with self.assertRaises(ValueError): scan_jaffe(root)
            p.unlink()
            Path(root, 'KA.AN1.39.tiff').touch()
            Path(root, 'copies').mkdir()
            Path(root, 'copies/KA-AN1.png').touch()
            with self.assertRaises(ValueError): scan_jaffe(root)

    def test_dataset_views_hide_truth_and_share_ids(self):
        # Exercise the actual dataset class without requiring GPU/torch imports.
        tree = ast.parse(Path('dataset.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'JAFFE')
        cv = SimpleNamespace(IMREAD_COLOR=1, COLOR_BGR2RGB=2,
                             imread=lambda *a: np.zeros((16, 16, 3), dtype=np.uint8),
                             cvtColor=lambda x, _: x)
        env = dict(data=SimpleNamespace(Dataset=object), scan_jaffe=scan_jaffe, cv2=cv)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'dataset.py', 'exec'), env)
        J = env['JAFFE']
        with TemporaryDirectory() as root:
            Path(root, 'KM.DI1.20.tiff').touch()
            transform = lambda x: x.shape
            adapt = J(root, transform=transform, weak2_transform=transform,
                      strong_transform=transform, return_index=True)
            evaluate = J(root, phase='eval', transform=transform)
            self.assertEqual(adapt.file_paths, evaluate.file_paths)
            self.assertEqual(adapt[0], ((16, 16, 3),) * 3 + (-1, 0))
            self.assertEqual(evaluate[0][1], 2)


if __name__ == '__main__':
    unittest.main()
