import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cast_v6.ccdr import class_volume_weights, classifier_modulation_loss
from cast_v6.ddrl import ddrl_loss, mmd2
from cast_v6.model import EMATeacher
from cast_v6.pseudo import build_pseudo_bank, source_class_correction


def test_mmd_finite_and_small_for_identical_sets():
    torch.manual_seed(0)
    x = torch.randn(16, 32)
    y = x.clone()
    value = mmd2(x, y)
    assert value is not None
    assert torch.isfinite(value)
    assert abs(float(value)) < 1.0


def test_ddrl_handles_missing_target_classes():
    torch.manual_seed(0)
    sf = torch.randn(24, 16)
    sy = torch.tensor([0] * 8 + [1] * 8 + [2] * 8)
    tf = torch.randn(12, 16)
    ty = torch.tensor([0] * 6 + [2] * 6)
    eta = torch.ones(7)
    loss, info = ddrl_loss(sf, sy, tf, ty, eta, num_classes=7, min_class_samples=2)
    assert torch.isfinite(loss)
    assert info["active_classes"] == 2
    assert set(info["intra_classes"]) == {0, 2}


def test_ccdr_volume_weights_are_finite():
    torch.manual_seed(0)
    feat = torch.randn(30, 8)
    labels = torch.tensor([0] * 10 + [1] * 10 + [3] * 10)
    eta, info = class_volume_weights(feat, labels, 7)
    assert torch.isfinite(eta).all()
    assert eta.min() > 0
    assert len(info["eta"]) == 7


def test_classifier_modulation_loss_finite():
    w = torch.randn(7, 32, requires_grad=True)
    loss, mean_cos = classifier_modulation_loss(w)
    assert torch.isfinite(loss)
    loss.backward()
    assert w.grad is not None
    assert math.isfinite(mean_cos)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x):
        z = self.bn(self.fc(x))
        return z, x


def test_ema_copies_buffers_exactly():
    torch.manual_seed(0)
    student = TinyNet()
    teacher = EMATeacher.from_student(student, decay=0.999)
    student.bn.running_mean.add_(1.0)
    teacher.update(student)
    assert torch.allclose(teacher.model.bn.running_mean, student.bn.running_mean)
    assert torch.equal(teacher.model.bn.num_batches_tracked, student.bn.num_batches_tracked)


class TinyPseudoDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([
            [4.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 3.5, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 3.5],
        ])
        self.y = torch.tensor([0, 0, 1, 1, 2, 2])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.x[i], self.y[i], i


class IdentityTeacher(nn.Module):
    def eval(self):
        return self

    def forward(self, x):
        return x, x


def test_pseudo_bank_catm_and_agreement():
    ds = TinyPseudoDataset()
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    bank = build_pseudo_bank(
        IdentityTeacher(), loader, len(ds), 3, torch.device("cpu"),
        epoch=0, total_epochs=30, phi=1.0, threshold_cap=0.99,
        debug_target_labels=True,
    )
    assert bank.agreement_rate == 1.0
    assert bank.pseudo_accuracy == 1.0
    assert sum(bank.predicted_counts) == len(ds)
    assert torch.isfinite(bank.thresholds).all()
    assert bank.pseudo_precision == [1.0, 1.0, 1.0]
    assert bank.pseudo_recall is not None


def test_source_class_correction_only_uses_source_bias():
    # Class 1 is strongly under-predicted and should be boosted; class 0 is
    # over-predicted and should not be boosted.  No target distribution enters.
    corr = source_class_correction(
        class_total=[100, 100, 100],
        predicted_counts=[160, 20, 120],
        alpha=0.5,
        min_correction=0.85,
        max_correction=1.5,
    )
    assert len(corr) == 3
    assert corr[1] > 1.0
    assert corr[0] <= 1.0
    assert all(0.85 <= x <= 1.5 for x in corr)
