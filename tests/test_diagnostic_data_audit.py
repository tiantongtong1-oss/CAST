import tempfile
import unittest
from pathlib import Path
from diagnostic_data_audit import assert_disjoint_paths,audit_raf_labels,EXPECTED_FER_TO_CAST

class DatasetAuditTests(unittest.TestCase):
    def test_path_overlap_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory(dir='.') as root:
            p=Path(root)/'a.jpg'
            with self.assertRaises(ValueError):assert_disjoint_paths([p],[p])
            with self.assertRaises(ValueError):assert_disjoint_paths([p,p],[Path(root)/'b.jpg'])
            self.assertEqual(assert_disjoint_paths([p],[Path(root)/'b.jpg'])['resolved_path_overlap'],0)

    def test_official_split_and_shuffled_alignment(self):
        with tempfile.TemporaryDirectory(dir='.') as root:
            root=Path(root);(root/'EmoLabel').mkdir()
            (root/'EmoLabel/list_patition_label.txt').write_text('train_1.jpg 1\ntrain_2.jpg 2\ntest_1.jpg 7\n')
            paths=[root/'train_2_aligned.jpg',root/'train_1_aligned.jpg']
            self.assertEqual(audit_raf_labels(root,paths,[1,0],'train')['official_samples'],2)
            with self.assertRaises(ValueError):audit_raf_labels(root,paths,[0,1],'train')
            with self.assertRaises(ValueError):audit_raf_labels(root,paths[:1],[1],'train')
            self.assertEqual(audit_raf_labels(root,[root/'test_1_aligned.jpg'],[6],'test')['official_samples'],1)

    def test_fer_mapping_is_bijection(self):
        self.assertEqual(set(EXPECTED_FER_TO_CAST.values()),set(range(7)))
        self.assertEqual(EXPECTED_FER_TO_CAST[2],1)

if __name__=='__main__':unittest.main()
