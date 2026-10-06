import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from source_knn_audit import SourceKNNAudit
from image_audit import ImageAudit


class SourceAuditTests(unittest.TestCase):
    def test_true_label_neighbors_and_export(self):
        features = np.repeat(np.eye(7, dtype=np.float32), 2, axis=0)
        labels = np.repeat(np.arange(7), 2)
        bank = SourceKNNAudit(features, labels, ['source%d.jpg' % i for i in range(14)], k=2)
        diagnostics, neighbors = bank.query(np.eye(7)[[5, 2]], [2, 2])
        self.assertEqual(diagnostics[0]['source_support_angry'], 1)
        self.assertFalse(diagnostics[0]['source_knn_agrees'])
        self.assertEqual(diagnostics[1]['source_support_disgust'], 1)
        self.assertTrue(diagnostics[1]['balanced_source_knn_agrees'])
        self.assertEqual(neighbors[0][0]['source_true'], 5)
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'original.jpg'; p.write_bytes(b'original bytes')
            audit = ImageAudit([str(p), str(p)])
            audit.add([5, 2], [2, 2], [.9, .8], [True, False],
                      probabilities=[[0,0,1,0,0,0,0]] * 2,
                      diagnostics=diagnostics, neighbors=neighbors)
            root = Path(tmp) / 'out'
            audit.save(root, 'checkpoint', source_metadata={'k': 2})
            with (root/'angry_to_disgust/samples.csv').open(encoding='utf-8-sig') as f:
                row = next(csv.DictReader(f))
            self.assertEqual(float(row['p_disgust']), 1)
            self.assertEqual(float(row['source_support_angry']), 1)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['angry_to_disgust']['source_agreement_fraction'], 0)
            self.assertEqual(summary['disgust_correct']['source_agreement_fraction'], 1)
            self.assertEqual(next(root.rglob('*.jpg')).read_bytes(), b'original bytes')

    def test_balanced_pool_and_small_k(self):
        labels = np.array([0]*5 + list(range(1,7)))
        bank = SourceKNNAudit(np.eye(7)[labels], labels, ['p']*len(labels), k=20)
        self.assertEqual(np.bincount(labels[bank.balanced_ids]).tolist(), [1]*7)
        rows, _ = bank.query(np.eye(7)[[2]], [2])
        self.assertAlmostEqual(sum(rows[0]['balanced_source_support_'+n] for n in
                                   ['surprise','fear','disgust','happy','sad','angry','neutral']), 1)

if __name__ == '__main__':
    unittest.main()
