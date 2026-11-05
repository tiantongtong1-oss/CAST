import ast
from pathlib import Path
import unittest
try:
    import torch
    from target_knn import TargetKNNBank
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class TargetKNNTests(unittest.TestCase):
    def bank(self, features, labels, ids, k=11):
        bank = TargetKNNBank(2, 2, k=k)
        bank.set_target_memory(torch.tensor(features, dtype=torch.float32),
                               torch.tensor(labels), torch.tensor(ids))
        return bank

    def query(self, bank, label=0, sample_id=10, confidence=True, agreement=True):
        return bank.select(torch.tensor([[1., 0.]]), torch.tensor([label]),
                           torch.tensor([confidence]), torch.tensor([agreement]),
                           torch.tensor([sample_id]))

    def test_exclude_self_by_id_and_vote(self):
        bank = self.bank([[1., 0.], [2., 0.], [3., 0.], [4., 0.]],
                         [0, 1, 1, 1], [10, 11, 12, 13], k=3)
        passed, checked = self.query(bank)
        self.assertTrue(checked.item())
        self.assertFalse(passed.item())
        passed, _ = self.query(bank, label=1)
        self.assertTrue(passed.item())

    def test_tie_rejected_with_self_excluded(self):
        bank = self.bank([[1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.]],
                         [0, 0, 0, 1, 1], [10, 11, 12, 13, 14])
        passed, checked = self.query(bank)
        self.assertTrue(checked.item())
        self.assertFalse(passed.item())

    def test_insufficient_other_neighbors_and_candidates(self):
        bank = self.bank([[1., 0.], [2., 0.], [3., 0.]], [0, 0, 0], [10, 11, 12])
        passed, checked = self.query(bank)
        self.assertFalse(passed.item())
        self.assertFalse(checked.item())
        bank = self.bank([[1., 0.], [2., 0.], [3., 0.], [4., 0.]],
                         [0, 0, 0, 0], [10, 11, 12, 13])
        for confidence, agreement in [(False, True), (True, False)]:
            passed, checked = self.query(bank, confidence=confidence, agreement=agreement)
            self.assertFalse(passed.item())
            self.assertFalse(checked.item())

    def test_invalid_memory_and_duplicate_ids(self):
        bank = self.bank([[0., 0.], [float('nan'), 1.], [1., 0.]],
                         [0, 0, 1], [10, 11, 12])
        self.assertEqual(bank.memory_ids.tolist(), [12])
        with self.assertRaises(ValueError):
            self.bank([[1., 0.], [2., 0.]], [0, 1], [10, 10])


class TargetIntegrationTests(unittest.TestCase):
    def test_no_radius_and_target_predictions_only(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'train.py').read_text()
        tree = ast.parse(source)
        self.assertNotIn('RawMedoidQuantileBank', source)
        self.assertNotIn('exact_medoid_index', source)
        self.assertNotIn('enable_radius_rescue', source)
        self.assertIn('agreement_mask, sample_ids,', source)
        self.assertIn('target_train.file_paths != target_threshold.file_paths', source)
        scan = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'calculate_target_thresholds')
        code = ast.get_source_segment(source, scan)
        self.assertIn('logits1.argmax(dim=1).cpu()', code)
        self.assertIn('for original, augmented, _, sample_ids in loader', code)
        self.assertIn('feature_hook.output.detach().cpu()', code)


if __name__ == '__main__':
    unittest.main()
