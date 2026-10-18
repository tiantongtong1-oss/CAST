import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from jaffe_manifest import scan_jaffe, POSE_TO_CAST
from jaffe_split import create_split, select_split


class SplitTests(unittest.TestCase):
    def test_fixed_partition_and_invalid_manifest(self):
        with TemporaryDirectory() as root:
            for (pose, label), count in zip(POSE_TO_CAST.items(), [30,32,29,31,31,30,30]):
                for i in range(count):
                    subject = 'K' + chr(65 + i // 4)
                    Path(root, '%s.%s%d.tiff' % (subject, pose, i % 4 + 1)).touch()
            obj = create_split(root)
            self.assertEqual(obj, create_split(root))
            self.assertNotEqual(obj, create_split(root, 42))
            manifest = Path(root, 'split.json')
            manifest.write_text(json.dumps(obj))
            records = scan_jaffe(root)
            train = select_split(root, records, manifest, 'adapt')
            test = select_split(root, records, manifest, 'eval')
            self.assertEqual((len(train), len(test)), (170,43))
            self.assertFalse(set(p for p,_ in train) & set(p for p,_ in test))
            self.assertEqual([sum(y==c for _,y in test) for c in range(7)], [6,7,6,6,6,6,6])
            obj['test'][0] = obj['train'][0]
            manifest.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                select_split(root, records, manifest, 'adapt')
