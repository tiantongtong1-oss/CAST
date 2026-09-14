from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torchvision import models


CLASS_NAMES = ("surprise", "fear", "disgust", "happy", "sad", "angry", "neutral")


class FERNet(nn.Module):
    """Shared FER backbone + classifier.

    The module names intentionally match the public CAST implementation
    (`feature`, `fc`, `bn`) so source checkpoints can be loaded directly.
    The classifier-output BatchNorm is kept for checkpoint compatibility,
    but target adaptation freezes its running statistics by default.
    """

    def __init__(self, backbone: str = "resnet50", num_classes: int = 7,
                 pretrained: bool = False, drop_rate: float = 0.5,
                 use_logit_bn: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.use_logit_bn = use_logit_bn
        self.bn = nn.BatchNorm1d(num_classes)

        if backbone == "resnet18":
            base = models.resnet18(pretrained=pretrained)
            self.feature = nn.Sequential(
                *list(base.children())[:-1], nn.Flatten(), nn.Dropout(drop_rate)
            )
            feat_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(pretrained=pretrained)
            self.feature = nn.Sequential(
                *list(base.children())[:-1],
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(2048, 512),
            )
            feat_dim = 512
        elif backbone == "mobilenet_v2":
            base = models.mobilenet_v2(pretrained=pretrained)
            self.feature = nn.Sequential(
                *list(base.children())[:-1],
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(1280, 512),
                nn.Dropout(drop_rate),
            )
            feat_dim = 512
        else:
            raise ValueError("Unsupported backbone: %s" % backbone)

        self.feature_dim = feat_dim
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.fc(features)
        if self.use_logit_bn:
            logits = self.bn(logits)
        return logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        logits = self.classify(features)
        return logits, features


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    if all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_source_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device,
                           min_coverage: float = 0.90) -> Dict[str, object]:
    raw = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    state = _strip_module_prefix(state)

    own = model.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in own and tuple(own[k].shape) == tuple(v.shape)
    }
    coverage = len(compatible) / max(1, len(own))
    if coverage < min_coverage:
        missing = sorted(set(own) - set(compatible))[:20]
        unexpected = sorted(set(state) - set(compatible))[:20]
        raise RuntimeError(
            "Checkpoint coverage %.1f%% < %.1f%%. Missing(sample)=%s unexpected(sample)=%s"
            % (coverage * 100.0, min_coverage * 100.0, missing, unexpected)
        )
    result = model.load_state_dict(compatible, strict=False)
    return {
        "coverage": coverage,
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "checkpoint": raw,
    }


def iter_batchnorm(module: nn.Module) -> Iterable[nn.modules.batchnorm._BatchNorm]:
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            yield m


def freeze_bn_stats(model: nn.Module, freeze_affine: bool = False,
                    freeze_logit_bn_only: bool = False) -> None:
    """Freeze running statistics while optionally leaving affine params trainable."""
    for name, m in model.named_modules():
        if not isinstance(m, nn.modules.batchnorm._BatchNorm):
            continue
        if freeze_logit_bn_only and name != "bn":
            continue
        m.eval()
        if freeze_affine and m.affine:
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)


def copy_bn_buffers(src: nn.Module, dst: nn.Module) -> None:
    src_modules = dict(src.named_modules())
    for name, d in dst.named_modules():
        if not isinstance(d, nn.modules.batchnorm._BatchNorm):
            continue
        s = src_modules.get(name)
        if s is None:
            continue
        if s.running_mean is not None and d.running_mean is not None:
            d.running_mean.copy_(s.running_mean)
        if s.running_var is not None and d.running_var is not None:
            d.running_var.copy_(s.running_var)
        if hasattr(s, "num_batches_tracked") and hasattr(d, "num_batches_tracked"):
            d.num_batches_tracked.copy_(s.num_batches_tracked)


@torch.no_grad()
def recalibrate_backbone_bn(model: FERNet, loader, device: torch.device,
                            batches: int = 64, momentum: float = 0.03) -> int:
    """Recalibrate only convolutional backbone BN buffers on target weak views."""
    if batches <= 0:
        return 0

    original_training = model.training
    original = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            original.append((m, m.training, m.momentum))
            m.train()
            m.momentum = momentum
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

    model.eval()
    for m, _, _ in original:
        m.train()

    seen = 0
    for batch in loader:
        if seen >= batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device, non_blocking=True)
        _ = model(x)
        seen += 1

    for m, train_flag, old_momentum in original:
        m.momentum = old_momentum
        m.train(train_flag)
    model.train(original_training)
    return seen


@dataclass
class EMATeacher:
    model: FERNet
    decay: float = 0.999
    step: int = 0

    @classmethod
    def from_student(cls, student: FERNet, decay: float = 0.999) -> "EMATeacher":
        teacher = copy.deepcopy(student)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return cls(model=teacher, decay=decay, step=0)

    @torch.no_grad()
    def update(self, student: FERNet) -> float:
        self.step += 1
        d = min(self.decay, 1.0 - 1.0 / float(self.step + 1))
        for t, s in zip(self.model.parameters(), student.parameters()):
            t.mul_(d).add_(s.detach(), alpha=1.0 - d)
        for tb, sb in zip(self.model.buffers(), student.buffers()):
            tb.copy_(sb.detach())
        self.model.eval()
        return d
