import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from sklearn.neighbors import KernelDensity

import Networks
import train
from dataset import FER, RafDataSet
from training_utils import (classification_losses, select_pseudo_labels,
                            target_affinity_weight, update_ema)


CASTModel = Networks.Model
torch.set_num_threads(1)


@pytest.mark.parametrize('maximum', [0.0, 0.003, 0.03])
def test_affinity_schedule_obeys_configured_maximum(maximum):
    values = [target_affinity_weight(epoch, maximum) for epoch in range(30)]
    assert all(0 <= value <= maximum for value in values)
    assert values[:5] == [0.0] * 5
    assert values[-1] == maximum


def test_both_views_must_be_confident_and_agree():
    p1 = torch.tensor([[0.99, 0.005, 0.005]] * 3)
    p2 = torch.tensor([[0.60, 0.20, 0.20], [0.98, 0.01, 0.01], [0.01, 0.98, 0.01]])
    labels, mask, weights, thresholds, agreement = select_pseudo_labels(
        p1.log(), p2.log(), 0, 30, 1.4)
    assert agreement.tolist() == [True, True, False]
    assert mask.tolist() == [False, True, False]
    assert weights[0] == weights[2] == 0
    assert 0 < weights[1] <= 1
    assert thresholds[2].item() == pytest.approx(0.95)  # missing class


def test_singleton_pseudo_batch_retains_dimension():
    logits = torch.tensor([[10., 0., 0.]])
    labels, mask, weights, _, _ = select_pseudo_labels(logits, logits, 29, 30, 1.4)
    assert labels.shape == mask.shape == weights.shape == (1,)
    assert mask.item()


def test_no_pseudo_labels_has_zero_target_gradient():
    logits = torch.randn(6, 7, requires_grad=True)
    source, target = classification_losses(logits, torch.tensor([0, 1]),
                                          torch.tensor([2, 3, 4, 5]), torch.zeros(4))
    (source + target).backward()
    assert target.item() == 0
    assert torch.count_nonzero(logits.grad[2:]) == 0
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:2]) > 0


def test_target_reduction_does_not_shrink_when_rejected_samples_are_added():
    logits = torch.randn(4, 7)
    args = (torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([0.2, 0.8]))
    _, expected = classification_losses(logits, *args)
    padded = torch.cat([logits, torch.randn(8, 7)])
    _, actual = classification_losses(padded, args[0], torch.cat([args[1], torch.zeros(8, dtype=torch.long)]),
                                     torch.cat([args[2], torch.zeros(8)]))
    torch.testing.assert_close(actual, expected)


def test_selected_target_weights_control_gradients():
    logits = torch.zeros(4, 7, requires_grad=True)
    _, target = classification_losses(logits, torch.tensor([0, 1]),
                                     torch.tensor([2, 3]), torch.tensor([0.25, 0.75]))
    target.backward()
    assert logits.grad[:2].abs().sum().item() == 0
    assert logits.grad[3].abs().sum().item() == pytest.approx(3 * logits.grad[2].abs().sum().item())
    assert logits.grad[2, 2] < 0 and logits.grad[3, 3] < 0


def test_real_resnet50_cpu_forward_backward():
    model = CASTModel(backbone='resnet50', num_classes=7, pretrained=False)
    logits, affinity = model(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]),
                             None, task='source', compute_affinity=False)
    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert logits.shape == (2, 7)
    assert affinity.item() == 0
    assert torch.isfinite(model.feature[0].weight.grad).all()
    assert torch.isfinite(model.fc.weight.grad).all()


def test_ema_updates_parameters_bn_statistics_and_integer_counter():
    student = nn.BatchNorm1d(3)
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    with torch.no_grad():
        student.weight.fill_(3)
        student.running_mean.fill_(4)
        student.num_batches_tracked.fill_(7)
    update_ema(student, teacher, 0.5)
    torch.testing.assert_close(teacher.weight, torch.full((3,), 2.))
    torch.testing.assert_close(teacher.running_mean, torch.full((3,), 2.))
    assert teacher.num_batches_tracked.item() == 7
    assert teacher.weight.grad is None


def test_missing_classes_do_not_zero_present_class_affinity_weights():
    weights = Networks.cal_weight(np.array([0., 0.5, 1., np.inf]))
    assert np.isfinite(weights).all()
    assert weights[[0, 3]].sum() == 0
    assert weights[1] > weights[2] > 0
    assert weights.sum() == pytest.approx(1.0)


def test_identical_features_and_empty_class_mmd_are_finite():
    source = torch.zeros(2, 3, requires_grad=True)
    target = torch.zeros(1, 3, requires_grad=True)
    loss = Networks.mmd_loss(source, target) + Networks.mmd_loss(source, target[:0])
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(source.grad).all()
    assert torch.isfinite(target.grad).all()


def tiny_model(backbone=None, num_classes=7, pretrained=False):
    # Use the real CAST forward/affinity implementation with cheap features.
    model = CASTModel.__new__(CASTModel)
    nn.Module.__init__(model)
    model.num_classes = num_classes
    model.feature = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
    model.fc = nn.Linear(8, num_classes, bias=False)
    model.bn = nn.BatchNorm1d(num_classes)
    model.kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
    return model


def test_zero_affinity_never_calls_kde(monkeypatch):
    model = tiny_model()
    def fail(*args):
        raise AssertionError('disabled affinity must not run')
    monkeypatch.setattr(model, 'volume', fail)
    logits, loss = model(torch.randn(4, 3, 8, 8), torch.arange(4), None,
                         compute_affinity=False)
    (logits.square().mean() + loss).backward()
    assert loss.item() == 0
    assert model.fc.weight.grad is not None


def test_domain_split_precedes_selection_for_unequal_batches(monkeypatch):
    model = tiny_model()
    model.feature = nn.Identity()
    calls = []
    original = model.split_feature_makeLD
    def record(features, targets):
        calls.append(features.detach().clone())
        return original(features, targets)
    monkeypatch.setattr(model, 'split_feature_makeLD', record)
    features = torch.arange(48).reshape(6, 8).float()
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    # Two source samples (only one retained), four targets (one retained).
    mask = torch.tensor([1, 0, 0, 0, 1, 0])
    logits, loss = model(features, labels, mask, source_count=2)
    torch.testing.assert_close(calls[0], features[[0]])
    torch.testing.assert_close(calls[1], features[[4]])
    (logits.square().mean() + loss * 0.01).backward()
    assert torch.isfinite(model.fc.weight.grad).all()


def make_images(tmp_path):
    source, target = tmp_path / 'raf', tmp_path / 'fer'
    (source / 'EmoLabel').mkdir(parents=True)
    (source / 'Image/aligned').mkdir(parents=True)
    records = []
    for i in range(14):
        name = 'train_%04d' % i
        records.append('%s.jpg %d\n' % (name, i % 7 + 1))
        image = np.random.RandomState(i).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(source / 'Image/aligned' / (name + '_aligned.jpg')), image)
        for split in ['train', 'test']:
            folder = target / split / str(i % 7)
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / ('%d.jpg' % i)), image)
    (source / 'EmoLabel/list_patition_label.txt').write_text(''.join(records))
    return source, target


def test_dataset_views_start_from_raw_rgb_and_mapping_is_stable(tmp_path):
    source, target = make_images(tmp_path)
    calls = []
    def weak(image):
        assert image.dtype == np.uint8
        return torch.full((3, 8, 8), -2.0)
    def strong(image):
        assert isinstance(image, np.ndarray) and image.dtype == np.uint8
        calls.append(image.copy())
        return torch.zeros(3, 8, 8)
    raf = RafDataSet(str(source), 'train', weak, strong)
    raf[0]
    fer = FER(str(target), 'train', weak, strong)
    assert len(fer[0]) == 4
    assert len(calls) == 2
    mapping = {0: 5, 1: 2, 2: 1, 3: 3, 4: 4, 5: 0, 6: 6}
    assert all(y == mapping[int(Path(p).parent.name)] for p, y in zip(fer.file_paths, fer.label))
    # Dataset construction must not reset the caller's global RNG.
    np.random.seed(17)
    expected = np.random.rand()
    np.random.seed(17)
    FER(str(target), 'train', weak, strong)
    assert np.random.rand() == expected


def test_weak_teacher_transform_has_no_erasing_crop_or_rotation():
    names = [type(t).__name__ for t in train.build_transforms()['weak'].transforms]
    assert not {'RandomErasing', 'RandomRotation', 'RandomCrop'}.intersection(names)


def test_evaluation_loss_weights_partial_batches_by_sample():
    class FixedModel(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x, x
    logits = torch.tensor([[3., 0., 0., 0., 0., 0., 0.]] * 2 + [[0., 3., 0., 0., 0., 0., 0.]])
    labels = torch.zeros(3, dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(logits, labels), batch_size=2)
    result = train.evaluate_only(FixedModel(), loader, torch.device('cpu'))
    assert result['accuracy'] == pytest.approx(2 / 3)
    assert result['loss'] == pytest.approx(nn.functional.cross_entropy(logits, labels).item())
    assert result['num_samples'] == 3


def test_source_target_smoke_and_checkpoint_reuse(tmp_path, monkeypatch):
    source, target = make_images(tmp_path)
    monkeypatch.setattr(train.Networks, 'Model', tiny_model)
    base = ['--source_root', str(source), '--target_root', str(target), '--output_dir', str(tmp_path / 'runs'),
            '--device', 'cpu', '--no_pretrained', '--workers', '0', '--batch_size', '7',
            '--source_epochs', '1', '--epochs', '2', '--w2', '0', '--target_w2', '0']
    best = train.run_training(train.parse_args(base + ['--run_name', 'first']))
    run_dir = tmp_path / 'runs/first'
    history = [json.loads(line) for line in (run_dir / 'history.jsonl').read_text().splitlines()]
    assert len([row for row in history if row['stage'] == 'target']) == 4
    assert {row['model_kind'] for row in history if row['stage'] == 'target'} == {'student', 'ema'}
    baseline = next(row['accuracy'] for row in history if row['stage'] == 'initial')
    assert best['overall'] >= baseline
    for name in ['source_best.pth', 'target_student_best.pth', 'target_ema_best.pth', 'best.pth']:
        saved = torch.load(run_dir / name, map_location='cpu')
        tiny_model().load_state_dict(saved['model'], strict=True)
        assert 'metrics' in saved and 'args' in saved and 'scheduler' in saved
    with pytest.raises(FileExistsError):
        train.run_training(train.parse_args(base + ['--run_name', 'first']))
    # The new flag bypasses all source epochs; legacy model/optimizer format still loads.
    legacy_path = tmp_path / 'legacy.pth'
    checkpoint = torch.load(run_dir / 'source_best.pth', map_location='cpu')
    torch.save({'model': checkpoint['model'], 'optimizer': checkpoint['optimizer']}, legacy_path)
    train.run_training(train.parse_args(base + ['--run_name', 'reuse', '--source_checkpoint', str(legacy_path)]))
    reuse_history = [json.loads(line) for line in (tmp_path / 'runs/reuse/history.jsonl').read_text().splitlines()]
    assert not any(row['stage'] == 'source' for row in reuse_history)
    # Identical source weights + stage seeds should reproduce target updates and metrics.
    expected = [row for row in history if row['stage'] != 'source']
    assert reuse_history == expected
