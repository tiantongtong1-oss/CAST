"""Exercise prototype writes through the real training loop with controlled gates."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import train
from test_v8_training import TinyData, TinyModel


class ImmediatePrototypeTests(unittest.TestCase):
    def test_cli_opt_in(self):
        self.assertFalse(train.parse_args([]).immediate_knn_prototype_update)
        self.assertTrue(train.parse_args(
            ['--immediate_knn_prototype_update']).immediate_knn_prototype_update)

    def test_rescue_updates_without_promoting_or_changing_mmd(self):
        original_gate = train.SparseReliableKNNBank.gate

        def controlled_gate(bank, *args, **kwargs):
            result = original_gate(bank, *args, **kwargs)
            result['pass_mask'][:] = True
            result['pass_mask'][-1] = False
            return result

        def controlled_confidence(logits1, logits2, thresholds):
            mask = torch.zeros(logits1.shape[0], device=logits1.device)
            mask[:2] = 1
            return logits1.argmax(1), mask, logits1.shape[0]

        for enabled, warmup, expected in ((False, 0, 0), (True, 0, 5), (True, 1, 0)):
            with self.subTest(enabled=enabled, warmup=warmup), tempfile.TemporaryDirectory() as root:
                masks = []

                class RecordingModel(TinyModel):
                    def forward(self, x, targets=None, idx=None, mode='test',
                                task=None, source_count=None):
                        if mode == 'train' and task == 'target':
                            masks.append(idx[source_count:].detach().clone())
                        return super().forward(x, targets, idx, mode, task, source_count)

                argv = ['train.py', '--device', 'cpu', '--workers', '0',
                        '--pre_epochs', '1', '--epochs', '1', '--promotion_epochs', '3',
                        '--knn_warmup_epochs', str(warmup), '--run_name', 'immediate_test']
                if enabled:
                    argv.append('--immediate_knn_prototype_update')
                output = io.StringIO()
                with patch('sys.argv', argv), patch.object(train, 'RafDataSet', TinyData), \
                     patch.object(train, 'FER', TinyData), \
                     patch.object(train.Networks, 'Model', RecordingModel), \
                     patch.object(train.util, 'make_confucion_matrix'), \
                     patch.object(train.SparseReliableKNNBank, 'gate', controlled_gate), \
                     patch.object(train, 'select_dual_view_pseudo_labels', controlled_confidence), \
                     contextlib.chdir(root), contextlib.redirect_stdout(output):
                    train.run_training()
                saved = torch.load(Path(root) / 'models/rafdb_fer/'
                                   'mobilenet_v2_rafdb_fer_immediate_test_target_best.pth',
                                   map_location='cpu', weights_only=False)
                bank = saved['prototype_bank']
                self.assertEqual(bank['target_counts'].sum().item(), expected)
                self.assertEqual(bank['target_initialized'].sum().item(), int(expected > 0))
                self.assertEqual(saved['args']['immediate_knn_prototype_update'], enabled)
                self.assertLess(saved['role_tracker']['streak'].max().item(), 3)
                self.assertTrue(masks)
                self.assertTrue(all(not mask.any() for mask in masks))
                self.assertIn('Immediate_Rescue_Update_Num: %d' % expected, output.getvalue())


if __name__ == '__main__':
    unittest.main()
