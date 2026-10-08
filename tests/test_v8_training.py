"""CPU integration: execute real training orchestration with tiny synthetic data.

No FER data, pretrained downloads or GPU are required. The real role/gate/loss/
checkpoint code runs; the image backbone and dataset are replaced for speed.
"""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

import train
from prototype_utils import FeatureHook
from reliability_roles import ReliabilityRoles
from sparse_reliable_knn import SparseReliableKNNBank


class TinyData(Dataset):
    def __init__(self, path, phase, weak2_transform=None, return_index=False, **kwargs):
        self.dual = weak2_transform is not None
        self.return_index = return_index
        self.file_paths = [str(i) for i in range(8)]

    def __len__(self):
        return 8

    def __getitem__(self, index):
        x = torch.tensor([1., (index - 3.5) * 0.005])
        y = 0
        if self.dual:
            return x, x.clone(), x.clone(), y, index
        if self.return_index:
            return x, y, index
        return x, y


class TinyModel(nn.Module):
    def __init__(self, backbone=None, num_classes=7):
        super().__init__()
        self.feature = nn.Linear(2, 2, bias=False)
        self.fc = nn.Linear(2, num_classes, bias=False)
        with torch.no_grad():
            self.feature.weight.copy_(torch.eye(2))
            self.fc.weight.zero_()
            self.fc.weight[0, 0] = 8.

    def forward(self, x, targets=None, idx=None, mode='test', task=None, source_count=None):
        f = self.feature(x)
        logits = self.fc(f)
        return logits, f.sum() * 0.0 if mode == 'train' else f


class TrainingTests(unittest.TestCase):
    def test_cli_defaults_and_invalid_combinations(self):
        args = train.parse_args([])
        self.assertTrue(args.role_separation)
        self.assertEqual(args.knn_score_mode, 'support')
        self.assertFalse(args.neighbor_soft_labels)
        for argv in (['--promotion_epochs', '0'], ['--knn_min_support', '21'],
                     ['--knn_min_purity', 'nan'], ['--knn_min_effective_support', 'nan'],
                     ['--knn_radius_multiplier', 'inf'], ['--run_name', '../bad'],
                     ['--neighbor_soft_labels', '--no_knn_gate'],
                     ['--neighbor_soft_labels', '--knn_score_mode', 'v7']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(argv)

    def test_refresh_ignores_target_labels_and_uses_promoted_senders(self):
        teacher = TinyModel(num_classes=2)
        hook = FeatureHook(teacher.feature)
        self.addCleanup(hook.close)
        x = torch.tensor([[1., -0.02], [1., 0.], [1., 0.02]])
        source = [(x, torch.zeros(3, dtype=torch.long))]
        roles = ReliabilityRoles(3, 2)
        for epoch in range(2):
            roles.observe(torch.arange(3), torch.zeros(3, dtype=torch.long),
                          torch.ones(3, dtype=torch.bool), torch.full((3,), 0.9), epoch)
        banks = []
        for fake_truth in (torch.tensor([999, -999, 3]), torch.tensor([1, 1, 1])):
            bank = SparseReliableKNNBank(2, 2)
            train.refresh_reliability_bank(teacher, hook, source, [(x, fake_truth, torch.arange(3))],
                                          bank, role_tracker=roles, epoch=2)
            banks.append(bank)
        self.assertTrue(torch.equal(banks[0].memory_sender_scores, banks[1].memory_sender_scores))
        self.assertTrue((banks[0].memory_sender_scores > 0).all())
        self.assertTrue(torch.equal(banks[0].memory_labels, torch.zeros(3, dtype=torch.long)))

    def test_full_cpu_training_and_checkpoint_for_default_and_soft_option(self):
        # Both routes execute through the public entry point. Soft-repair gradients
        # on an actual conflict are covered in test_role_sparse_v8.py.
        for soft in (False, True):
            with self.subTest(soft=soft), tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
                argv = ['train.py', '--device', 'cpu', '--workers', '0',
                        '--pre_epochs', '1', '--epochs', '4', '--promotion_epochs', '2',
                        '--knn_warmup_epochs', '1', '--run_name', 'smoke']
                if soft:
                    argv.append('--neighbor_soft_labels')
                stdout = io.StringIO()
                with patch('sys.argv', argv), patch.object(train, 'RafDataSet', TinyData), \
                     patch.object(train, 'FER', TinyData), patch.object(train.Networks, 'Model', TinyModel), \
                     patch.object(train.util, 'make_confucion_matrix'), \
                     contextlib.chdir(root), contextlib.redirect_stdout(stdout):
                    accuracy = train.run_training()
                self.assertEqual(accuracy, 1.)
                self.assertIn('Stable_Senders: 8/8', stdout.getvalue())
                self.assertIn('UAR(present classes): 1.0000', stdout.getvalue())
                checkpoint_path = Path(root) / 'models/rafdb_fer/mobilenet_v2_rafdb_fer_smoke_target_best.pth'
                saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                self.assertIn('role_tracker', saved)
                self.assertNotIn('memory_sender_scores', saved['reliability_bank'])
                # Best epoch is the first tied maximum: no premature promotion.
                self.assertEqual(saved['prototype_bank']['target_counts'].sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
