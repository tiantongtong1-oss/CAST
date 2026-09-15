import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import FER
from ema_utils import select_dual_view_pseudo_labels
from temporal_utils import TemporalPredictionBank, mixed_target_cross_entropy


class TemporalTests(unittest.TestCase):
    def observe(self, bank, epoch, labels, indices=None, mask=None, warmup=0):
        labels = torch.tensor(labels)
        indices = torch.arange(len(labels)) if indices is None else torch.tensor(indices)
        logits = F.one_hot(labels, bank.probabilities.size(1)).float() * 5.0
        mask = torch.ones(len(labels)) if mask is None else torch.tensor(mask)
        return bank.update_and_select(indices, logits, logits, labels, mask, epoch, warmup)

    def test_warmup_streak_and_history_survive_shuffling(self):
        bank = TemporalPredictionBank(3, 3)
        mask, _, _ = self.observe(bank, 0, [0, 1, 2], warmup=2)
        self.assertEqual(mask.tolist(), [1, 1, 1])
        self.observe(bank, 1, [2, 0, 1], indices=[2, 0, 1], warmup=2)
        mask, probs, stats = self.observe(bank, 2, [1, 2, 0], indices=[1, 2, 0])
        self.assertEqual(mask.tolist(), [1, 1, 1])
        self.assertEqual(probs.argmax(1).tolist(), [1, 2, 0])
        self.assertEqual(stats['label_flips'], 0)
        self.assertEqual(bank.streaks.tolist(), [3, 3, 3])

    def test_current_prediction_cannot_validate_itself(self):
        bank = TemporalPredictionBank(1, 3, momentum=0.0, min_streak=1)
        mask, _, _ = self.observe(bank, 0, [0])
        self.assertEqual(mask.item(), 0)
        mask, probs, stats = self.observe(bank, 1, [1])
        self.assertEqual(mask.item(), 0)
        self.assertEqual(probs.argmax(1).item(), 1)
        self.assertEqual(stats['history_disagrees'], 1)
        mask, _, _ = self.observe(bank, 2, [1])
        self.assertEqual(mask.item(), 1)

    def test_rejected_or_missing_epoch_resets_streak(self):
        bank = TemporalPredictionBank(1, 3)
        self.observe(bank, 0, [1])
        self.observe(bank, 1, [1], mask=[0])
        mask, _, _ = self.observe(bank, 2, [1])
        self.assertEqual(mask.item(), 0)
        mask, _, _ = self.observe(bank, 3, [1])
        self.assertEqual(mask.item(), 1)
        mask, _, _ = self.observe(bank, 5, [1])
        self.assertEqual(mask.item(), 0)

    def test_old_wrong_label_can_recover(self):
        bank = TemporalPredictionBank(1, 3)
        self.observe(bank, 0, [0])
        masks = [self.observe(bank, epoch, [1])[0].item() for epoch in range(1, 6)]
        self.assertEqual(masks[0], 0)
        self.assertEqual(masks[-1], 1)
        self.assertEqual(bank.probabilities.argmax(1).item(), 1)

    def test_duplicate_visit_rejected(self):
        bank = TemporalPredictionBank(2, 3)
        with self.assertRaises(ValueError):
            self.observe(bank, 0, [0, 0], indices=[0, 0])
        self.observe(bank, 0, [0], indices=[0])
        with self.assertRaises(ValueError):
            self.observe(bank, 0, [0], indices=[0])

    def test_memory_checkpoint_roundtrip(self):
        bank = TemporalPredictionBank(2, 3)
        self.observe(bank, 0, [0, 1])
        buffer = io.BytesIO()
        torch.save(bank.state_dict(), buffer)
        buffer.seek(0)
        restored = TemporalPredictionBank(2, 3)
        restored.load_state_dict(torch.load(buffer, map_location='cpu'))
        first = self.observe(bank, 1, [0, 1])
        second = self.observe(restored, 1, [0, 1])
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_dual_view_remains_required(self):
        logits1 = torch.tensor([[8., 0., 0.], [8., 0., 0.], [8., 0., 0.]])
        logits2 = torch.tensor([[8., 0., 0.], [0., 8., 0.], [0.1, 0., 0.]])
        labels, mask, agreement = select_dual_view_pseudo_labels(
            logits1, logits2, torch.full((3,), 0.8))
        self.assertEqual(mask.tolist(), [1, 0, 0])
        self.assertEqual(agreement, 2)
        bank = TemporalPredictionBank(3, 3)
        for epoch in range(3):
            selected, _, _ = bank.update_and_select(
                torch.arange(3), logits1, logits2, labels, mask, epoch, 2)
        self.assertEqual(selected.tolist(), [1, 0, 0])

    def test_soft_ce_detaches_teacher_and_masks_gradients(self):
        logits = torch.tensor([[3., 0., -1.], [1., 2., 0.]], requires_grad=True)
        labels = torch.tensor([0, 1])
        teacher = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], requires_grad=True)
        hard = mixed_target_cross_entropy(logits, labels, teacher, 0.0)
        self.assertTrue(torch.equal(hard, F.cross_entropy(logits, labels, reduction='none')))
        losses = mixed_target_cross_entropy(logits, labels, teacher, 0.5)
        (losses * torch.tensor([1., 0.])).sum().backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)
        # A softened high-confidence target penalizes overconfident logits.
        self.assertGreater(logits.grad[0, 0].item(), 0)

    def test_empty_reliable_mask_has_finite_zero_gradient(self):
        logits = torch.randn(3, 7, requires_grad=True)
        loss = mixed_target_cross_entropy(
            logits, torch.zeros(3, dtype=torch.long), torch.full((3, 7), 1. / 7.))
        (loss * torch.zeros(3)).sum().backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)


class DatasetIndexTests(unittest.TestCase):
    def test_stable_paths_and_backward_compatible_tuple_shapes(self):
        import cv2
        with tempfile.TemporaryDirectory() as root:
            for label in (0, 1, 2):
                folder = Path(root) / 'train' / str(label)
                folder.mkdir(parents=True)
                cv2.imwrite(str(folder / 'b.jpg'), np.zeros((4, 4, 3), dtype=np.uint8))
            paths = [str(p) for p in Path(root).glob('train/*/*.jpg')]
            with redirect_stdout(io.StringIO()):
                with patch('dataset.glob.glob', return_value=paths):
                    legacy = FER(root, 'train')
                with patch('dataset.glob.glob', return_value=list(reversed(paths))):
                    indexed = FER(root, 'train', return_index=True)
            self.assertEqual(legacy.file_paths, indexed.file_paths)
            self.assertEqual(len(legacy[0]), 2)
            self.assertEqual(len(indexed[0]), 3)
            for i in range(len(indexed)):
                self.assertEqual(indexed[i][-1], i)
                original = int(Path(indexed.file_paths[i]).parent.name)
                self.assertEqual(indexed[i][-2], FER.FER_TO_CAST[original])
            identity = lambda image: image
            for weak2, strong, length in ((None, identity, 4),
                                          (identity, None, 4),
                                          (identity, identity, 5)):
                with redirect_stdout(io.StringIO()):
                    dataset = FER(root, 'train', weak2_transform=weak2,
                                  strong_transform=strong, return_index=True)
                self.assertEqual(len(dataset[0]), length)


class TrainingIntegrationTests(unittest.TestCase):
    """Run the actual orchestration on CPU with tiny synthetic input/model.

    Only device transfers, data/model factories and validation scores are
    patched. Real optimizer, CAST losses, hooks, banks, EMA and checkpoint IO
    exercise both stages. Impossible target train labels catch label leakage.
    """

    def run_case(self, source_score, student_score, ema_score, extra_args=(),
                 checkpoint=False, eval_ema=True, source_state=None, return_source=False):
        import train

        class TinyModel(train.Networks.Model):
            def __init__(self, **kwargs):
                nn.Module.__init__(self)
                self.num_classes = 7
                self.density_bandwidth = 0.2
                self.feature = nn.Sequential(nn.Flatten(), nn.Linear(12, 8))
                self.fc = nn.Linear(8, 7, bias=False)
                self.bn = nn.BatchNorm1d(7)

        class Source(torch.utils.data.Dataset):
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 14

            def __getitem__(self, index):
                return torch.full((3, 2, 2), (index + 1) / 14.), index % 7

        class Target(Source):
            def __init__(self, path, phase, return_index=False, **kwargs):
                self.phase, self.return_index = phase, return_index
                self.file_paths = [os.path.join(path, str(i) + '.jpg') for i in range(14)]

            def __getitem__(self, index):
                image, label = super().__getitem__(index)
                if self.return_index:
                    return image, image, image, 999, index
                return image, (999 if self.phase == 'train' else label)

        test_states = []
        student_trace = []
        original_load = torch.load

        def cpu_load(path, **kwargs):
            return original_load(path, map_location='cpu')

        def score(model, loader, criterion, size, epoch, split):
            if split == 'Test':
                test_states.append({key: value.clone() for key, value in model.state_dict().items()})
                return 0.42
            if split == 'Validation EMA':
                return ema_score
            if split == 'Validation Student':
                student_trace.append({key: value.clone() for key, value in model.state_dict().items()})
                return student_score
            return source_score

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as root:
            try:
                os.chdir(root)
                args = ['train.py', '--workers', '0', '--pre_epochs', '1', '--epochs', '3',
                        '--threshold_base', '0', '--threshold_beta', '0',
                        '--threshold_min', '0', '--threshold_max', '0', '--threshold_margin', '0',
                        '--proto_warmup_epochs', '0', '--proto_ramp_epochs', '0'] + list(extra_args)
                if eval_ema:
                    args += ['--eval_ema']
                if checkpoint:
                    if source_state is None:
                        model = TinyModel()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=1e-4)
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
                        checkpoint_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                            'scheduler': scheduler.state_dict()}
                    else:
                        checkpoint_state = source_state
                    torch.save(checkpoint_state, 'source.pth')
                    args += ['--pre_epochs', '0', '--checkpoint', 'source.pth']
                with patch('sys.argv', args), \
                     patch.object(train.Networks, 'Model', TinyModel), \
                     patch.object(train, 'RafDataSet', Source), \
                     patch.object(train, 'FER', Target), \
                     patch.object(torch.Tensor, 'cuda', lambda self, *a, **k: self), \
                     patch.object(nn.Module, 'cuda', lambda self, *a, **k: self), \
                     patch.object(train.torch, 'load', side_effect=cpu_load), \
                     patch.object(train, 'evaluate', side_effect=score), \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(train.run_training(), 0.42)
                base = Path('models/rafdb_fer/mobilenet_v2_rafdb_fer_prototype_recovery_v2')
                source = cpu_load(str(base) + '_source_best.pth')
                target = cpu_load(str(base) + '_target_best.pth')
                self.assertEqual(len(test_states), 1)
                expected = source['model'] if source_score >= max(student_score, ema_score) else (
                    target['ema_teacher'] if ema_score > student_score else target['model'])
                for key in expected:
                    self.assertTrue(torch.equal(test_states[0][key], expected[key]), key)
                if '--disable_temporal' not in extra_args:
                    self.assertTrue(target['temporal_bank']['seen'].all())
                    self.assertEqual(len(target['target_sample_paths']), 14)
                else:
                    self.assertNotIn('temporal_bank', target)
                if checkpoint and source_state is None:
                    self.assertEqual(source['optimizer']['param_groups'][0]['lr'], 0.0004)
                return (student_trace, source) if return_source else student_trace
            finally:
                os.chdir(original_cwd)

    def test_ema_is_loaded_when_validation_winner(self):
        self.run_case(0.2, 0.3, 0.4)

    def test_source_is_kept_when_adaptation_degrades(self):
        self.run_case(0.5, 0.3, 0.4)

    def test_student_winner_and_source_checkpoint_reuse(self):
        self.run_case(0.2, 0.4, 0.3, checkpoint=True)

    def test_baseline_ablation_without_temporal_or_soft_loss(self):
        self.run_case(0.2, 0.4, 0.3, extra_args=['--disable_temporal', '--target_soft_weight', '0'],
                      eval_ema=False)

    def test_observation_mode_does_not_change_training(self):
        observed = self.run_case(0.2, 0.4, 0.3)
        disabled = self.run_case(0.2, 0.4, 0.3, extra_args=['--disable_temporal'])
        self.assertEqual(len(observed), 3)
        for observed_epoch, disabled_epoch in zip(observed, disabled):
            for key in observed_epoch:
                self.assertTrue(torch.equal(observed_epoch[key], disabled_epoch[key]), key)

    def test_filter_and_soft_loss_remain_explicitly_available(self):
        self.run_case(0.2, 0.4, 0.3, extra_args=['--temporal_mode', 'filter',
                                              '--target_soft_weight', '0.5'])

    def test_optional_labeled_source_balancing_runs(self):
        self.run_case(0.2, 0.4, 0.3, extra_args=['--source_balance_power', '0.5'])

    def test_reusing_source_checkpoint_matches_full_source_then_target(self):
        full_trace, source = self.run_case(0.2, 0.4, 0.3, return_source=True)
        reused_trace = self.run_case(0.2, 0.4, 0.3, checkpoint=True, source_state=source)
        for full_epoch, reused_epoch in zip(full_trace, reused_trace):
            for key in full_epoch:
                self.assertTrue(torch.equal(full_epoch[key], reused_epoch[key]), key)

    def test_evaluation_loss_weights_samples_and_reports_recalls(self):
        import train

        class LogitModel(nn.Module):
            def forward(self, images, *args, **kwargs):
                return images, images

        logits = torch.tensor([[5., 0., 0., 0., 0., 0., 0.],
                               [0., 5., 0., 0., 0., 0., 0.],
                               [0., 0., 5., 0., 0., 0., 0.]])
        labels = torch.tensor([0, 1, 6])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(logits, labels), batch_size=2)
        expected_loss = F.cross_entropy(logits, labels).item()
        output = io.StringIO()
        with patch.object(torch.Tensor, 'cuda', lambda self, *a, **k: self), redirect_stdout(output):
            accuracy = train.evaluate(LogitModel(), loader,
                                      nn.CrossEntropyLoss(reduction='none'), 3, 0, 'Validation')
        self.assertAlmostEqual(accuracy, 2. / 3.)
        self.assertIn('Loss: %.3f' % expected_loss, output.getvalue())
        self.assertIn('Macro Recall: 0.2857', output.getvalue())


if __name__ == '__main__':
    unittest.main()
