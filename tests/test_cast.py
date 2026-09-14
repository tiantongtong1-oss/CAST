"""Behavioral regressions for the MobileNetV2 dual-view CAST training path."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

import Networks
from ema_utils import (PrototypeMemory, align_dual_view_probabilities,
                       create_ema_teacher, select_dual_view_pseudo_labels,
                       update_ema_teacher, weighted_mean_loss)
from train import (affinity_ramp, backward_and_step, calculate_teacher_statistics,
                   generate_dual_view_pseudo_labels, parse_args)


torch.set_num_threads(1)


class LogitTeacher(nn.Module):
    """Inputs are logits so filtering cases can be specified without a CNN."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, x, *unused, **kwargs):
        return x + self.anchor * 0, x


class CastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.args = parse_args([])
        self.args.teacher_temperature = 1.0
        self.args.distribution_power = 0.0
        self.prior = torch.ones(3) / 3

    def select(self, p1, p2, threshold=0.95):
        return select_dual_view_pseudo_labels(
            torch.tensor(p1), torch.tensor(p2), torch.full((3,), threshold),
            self.prior, self.args)

    def test_two_view_agreement_and_strict_fallback(self):
        p1 = [[.96, .02, .02], [.97, .02, .01], [.98, .01, .01], [.93, .04, .03]]
        p2 = [[.96, .02, .02], [.02, .97, .01], [.84, .08, .08], [.93, .04, .03]]
        _, mask, weights, agree = self.select(p1, p2)
        self.assertEqual(mask.tolist(), [1, 0, 0, 0])
        self.assertEqual(agree, 3)
        self.assertTrue(torch.equal(weights[1:], torch.zeros(3)))
        # Average .91 is insufficient when the second view is only .84.
        _, mask, _, _ = self.select(p1[2:], p2[2:])
        self.assertEqual(mask.tolist(), [0, 1])

    def test_no_eligible_fallback_stays_empty(self):
        _, mask, weights, _ = self.select([[.85, .10, .05]], [[.85, .10, .05]])
        self.assertEqual(mask.sum().item(), 0)
        self.assertEqual(weights.sum().item(), 0)

    def test_weight_bound_and_mean_normalization(self):
        self.args.pseudo_weight_max = .25
        _, _, weights, _ = self.select([[.99, .005, .005]], [[.99, .005, .005]])
        self.assertLessEqual(weights.max().item(), .25)
        losses = torch.tensor([2., 6., 100.], requires_grad=True)
        result = weighted_mean_loss(losses, torch.tensor([1., 3., 0.]))
        self.assertAlmostEqual(result.item(), 5.)
        result.backward()
        torch.testing.assert_close(losses.grad, torch.tensor([.25, .75, 0.]))

    def test_empty_weight_has_zero_gradient(self):
        logits = torch.randn(3, 3, requires_grad=True)
        loss = weighted_mean_loss(F.cross_entropy(logits, torch.zeros(3, dtype=torch.long),
                                                 reduction='none'), torch.zeros(3))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_teacher_and_bn_buffers_are_ema_only(self):
        student = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        teacher = create_ema_teacher(student)
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        before = teacher[0].weight.clone()
        with torch.no_grad():
            student[0].weight.add_(2)
            student[1].running_mean.fill_(4)
            student[1].num_batches_tracked.fill_(7)
        update_ema_teacher(teacher, student, .99, 1)
        torch.testing.assert_close(teacher[0].weight, before + 1)
        torch.testing.assert_close(teacher[1].running_mean, torch.full((3,), 2.))
        self.assertEqual(teacher[1].num_batches_tracked.item(), 7)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_statistics_match_per_view_correction_and_ignore_labels(self):
        teacher = LogitTeacher()
        self.args.distribution_power = .5
        self.args.phi = 1.
        self.args.threshold_min, self.args.threshold_max = 0., 1.
        self.args.threshold_stage_gain = 0.
        p1 = torch.tensor([[.99, .005, .005], [.3, .6, .1], [.7, .2, .1]])
        p2 = torch.tensor([[.4, .5, .1], [.9, .08, .02], [.3, .2, .5]])
        loader = [(p1.log(), p2.log(), torch.tensor([0, 1, 2]))]
        thresholds, prior = calculate_teacher_statistics(teacher, loader, 3, 0, 2, self.args)
        q1, q2 = align_dual_view_probabilities(p1, p2, prior, .5, 3.)
        conf, labels = ((q1 + q2) / 2).max(1)
        expected = torch.ones(3)
        for c in labels.unique():
            expected[c] = conf[labels == c].mean()
        torch.testing.assert_close(thresholds, expected)
        altered = [(p1.log(), p2.log(), torch.tensor([-999, 999, 123]))]
        other = calculate_teacher_statistics(teacher, altered, 3, 0, 2, self.args)
        torch.testing.assert_close(thresholds, other[0])
        torch.testing.assert_close(prior, other[1])
        generated = generate_dual_view_pseudo_labels(teacher, p1.log(), p2.log(),
                                                     thresholds, prior, self.args, True)
        self.assertFalse(generated[4].requires_grad)
        self.assertEqual(generated[4].device.type, 'cpu')

    def test_affinity_negative_term_changes_student_gradient(self):
        memory = PrototypeMemory(3, 2, torch.device('cpu'), margin=.2)
        memory.update(torch.tensor([[1., 0.], [.8, .6]]), torch.tensor([0, 1]))
        feature = torch.tensor([[.8, .6]], requires_grad=True)
        full = memory.loss(feature, torch.tensor([0]))
        grad_full, = torch.autograd.grad(full, feature)
        compact_feature = feature.detach().clone().requires_grad_()
        compact = 1 - F.normalize(compact_feature, dim=1)[0, 0]
        grad_compact, = torch.autograd.grad(compact, compact_feature)
        self.assertGreater((grad_full - grad_compact).abs().sum().item(), .01)
        self.assertGreater(full.item(), compact.item())
        self.assertFalse(memory.prototypes.requires_grad)

    def test_missing_classes_zero_weights_and_bad_memory_inputs(self):
        memory = PrototypeMemory(3, 2, torch.device('cpu'))
        memory.update(torch.tensor([[1., 0.]]), torch.tensor([0]), torch.zeros(1))
        memory.update(torch.tensor([[float('nan'), 1.], [0., 0.]]), torch.tensor([1, 2]))
        self.assertFalse(memory.initialized.any())
        x = torch.randn(2, 2, requires_grad=True)
        memory.loss(x, torch.tensor([0, 1])).backward()
        self.assertEqual(x.grad.abs().sum().item(), 0)
        memory.update(torch.tensor([[1., 0.]]), torch.tensor([0]))
        x2 = torch.tensor([[.8, .6], [.6, .8]], requires_grad=True)
        loss = memory.loss(x2, torch.tensor([0, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(x2.grad).all())
        self.assertEqual(x2.grad[1].abs().sum().item(), 0)
        restored = PrototypeMemory(3, 2, torch.device('cpu'))
        restored.load_state_dict(memory.state_dict())
        torch.testing.assert_close(memory.prototypes, restored.prototypes)
        torch.testing.assert_close(memory.initialized, restored.initialized)

    def test_affinity_warmup_and_upper_bound(self):
        weights = [affinity_ramp(e, 2, 5, .1) for e in range(12)]
        self.assertEqual(weights[:2], [0., 0.])
        self.assertAlmostEqual(weights[2], .02)
        self.assertEqual(weights[-1], .1)
        self.assertTrue(all(0 <= w <= .1 for w in weights))

    def test_mmd_matches_broadcast_reference_and_backpropagates(self):
        s, t = torch.randn(3, 5, requires_grad=True), torch.randn(4, 5, requires_grad=True)
        total = torch.cat((s, t))
        dist = ((total[:, None] - total[None, :]) ** 2).sum(2)
        bandwidth = dist.detach().sum() / 42 / 4
        expected = sum(torch.exp(-dist / (bandwidth * 2 ** i)) for i in range(5))
        actual = Networks.compute_kernel_matrix(s, t)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        Networks.mmd_loss(s, t).backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(t.grad.abs().sum().item(), 0)
        singleton = torch.ones(1, 5, requires_grad=True)
        Networks.mmd_loss(singleton, singleton).backward()
        self.assertEqual(singleton.grad.abs().sum().item(), 0)

    def test_ddrl_masks_original_source_target_boundary(self):
        model = Networks.Model(pretrained=False)
        model.feature = nn.Identity()
        model.eval()  # keep logits BN deterministic; mode still requests DDRL
        features = torch.randn(8, 512, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        mask = torch.tensor([0., 1., 1., 1., 1., 1., 1., 1.])
        out = model(features, labels, mask, source_count=4)
        # Equivalent manually compacted source + target partition.
        reference = model(features[mask.bool()], labels[mask.bool()], torch.ones(7), source_count=3)
        torch.testing.assert_close(out[3], reference[3])
        torch.testing.assert_close(out[4], reference[4])
        torch.testing.assert_close(out[1], out[3] + out[4])
        out[1].backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertEqual(features.grad[0].abs().sum().item(), 0)
        none = model(features, labels, torch.zeros(8), source_count=4)
        self.assertEqual(none[1].item(), 0)

    def test_mobilenet_shapes_and_target_backward(self):
        model = Networks.Model(pretrained=False, drop_rate=0.)
        model.train()
        out = model(torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 0, 1]),
                    torch.ones(4), source_count=2)
        self.assertEqual(out[0].shape, (4, 7))
        self.assertEqual(out[2].shape, (4, 512))
        (F.cross_entropy(out[0], torch.tensor([0, 1, 0, 1])) + out[1]).backward()
        self.assertTrue(torch.isfinite(model.fc.weight.grad).all())
        self.assertGreater(model.feature[0][0][0].weight.grad.abs().sum().item(), 0)

    def test_nonfinite_loss_and_gradients_do_not_step_optimizer(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        old = model.weight.clone()
        with self.assertRaises(FloatingPointError):
            backward_and_step(model.weight.sum() * float('nan'), model, optimizer, 5.)
        torch.testing.assert_close(old, model.weight)
        optimizer.zero_grad()
        with torch.no_grad():
            model.weight.zero_()
        # sqrt(0) is finite with an infinite derivative.
        with self.assertRaises(FloatingPointError):
            backward_and_step(model.weight.sqrt().sum(), model, optimizer, 5.)
        self.assertEqual(model.weight.item(), 0)

    def test_zero_pretraining_requires_explicit_checkpoint(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--pre_epochs', '0'])
        args = parse_args(['--pre_epochs', '0', '--checkpoint', 'source.pth'])
        self.assertEqual(args.checkpoint, 'source.pth')


class TrainingIntegrationTest(unittest.TestCase):
    def test_real_mobilenet_two_stages_and_checkpoint_start(self):
        """Tiny synthetic images validate plumbing, not FER accuracy."""
        import os
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, target = root / 'raf', root / 'fer'
            (source / 'EmoLabel').mkdir(parents=True)
            (source / 'Image/aligned').mkdir(parents=True)
            rng = np.random.default_rng(5)
            lines = []
            for c in range(7):
                name = 'train_%04d' % c
                lines.append('%s.jpg %d\n' % (name, c + 1))
                Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(
                    source / ('Image/aligned/%s_aligned.jpg' % name))
            (source / 'EmoLabel/list_patition_label.txt').write_text(''.join(lines))
            for split in ('train', 'val', 'test'):
                for c in range(3):
                    folder = target / split / str(c)
                    folder.mkdir(parents=True)
                    Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(folder / '0.jpg')
            base = [sys.executable, 'train.py', '--source_path', str(source), '--target_path', str(target),
                    '--device', 'cpu', '--no_pretrained', '--workers', '0', '--batch_size', '7',
                    '--eval_batch_size', '3', '--epochs', '1', '--aff_warmup_epochs', '0',
                    '--threshold_min', '0', '--threshold_max', '0', '--consistency_min_conf', '0',
                    '--distribution_power', '0', '--teacher_temperature', '1']
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            first = subprocess.run(base + ['--pre_epochs', '1', '--output_dir', str(root / 'first')],
                                   capture_output=True, text=True, env=env, timeout=180)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn('final target test accuracy', first.stdout)
            directory = root / 'first/rafdb_fer'
            source_checkpoint = next(directory.glob('*source_best.pth'))
            target_checkpoint = next(directory.glob('*target_best.pth'))
            state = torch.load(target_checkpoint, map_location='cpu')
            self.assertIn('ema_teacher', state)
            self.assertIn('prototypes', state)
            self.assertTrue(state['prototypes']['initialized'].all())
            self.assertAlmostEqual(state['optimizer']['param_groups'][0]['lr'], .001 * .95)
            second = subprocess.run(base + ['--pre_epochs', '0', '--checkpoint', str(source_checkpoint),
                                           '--output_dir', str(root / 'second'), '--target_lr', '.0002'],
                                    capture_output=True, text=True, env=env, timeout=180)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            target2 = next((root / 'second/rafdb_fer').glob('*target_best.pth'))
            state2 = torch.load(target2, map_location='cpu')
            self.assertAlmostEqual(state2['optimizer']['param_groups'][0]['lr'], .0002 * .95)


if __name__ == '__main__':
    unittest.main()
