"""Separate domain statistics with shared affine parameters and plain-BN exports.

Related work: Chang et al., Domain-Specific Batch Normalization for
Unsupervised Domain Adaptation, CVPR 2019. This variant shares affine weights.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DomainStatsMixin:
    def init_source_stats(self):
        self.register_buffer("source_running_mean", self.running_mean.detach().clone())
        self.register_buffer("source_running_var", self.running_var.detach().clone())
        self.register_buffer("source_num_batches_tracked", self.num_batches_tracked.detach().clone())
        self.domain = "target"
        self.freeze_target_stats = False

    def forward(self, x):
        self._check_input_dim(x)
        source = self.domain == "source"
        mean = self.source_running_mean if source else self.running_mean
        var = self.source_running_var if source else self.running_var
        count = self.source_num_batches_tracked if source else self.num_batches_tracked
        update = self.training and (source or not self.freeze_target_stats)
        # BN1d cannot estimate a variance from a singleton feature batch.
        update = update and x.numel() // x.shape[1] > 1
        momentum = self.momentum if self.momentum is not None else 0.0
        if update:
            count.add_(1)
            if self.momentum is None:
                momentum = 1.0 / float(count.item())
        return F.batch_norm(x, mean, var, self.weight, self.bias, update,
                            momentum, self.eps)


class DomainBatchNorm1d(DomainStatsMixin, nn.BatchNorm1d):
    pass


class DomainBatchNorm2d(DomainStatsMixin, nn.BatchNorm2d):
    pass


def enable_domain_bn(model, momentum=0.03):
    """Call after loading a plain source checkpoint, before building optimizer."""
    def convert(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, DomainStatsMixin):
                continue
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if not child.track_running_stats:
                    raise ValueError("Domain BN requires running statistics")
                cls = DomainBatchNorm1d if isinstance(child, nn.BatchNorm1d) else DomainBatchNorm2d
                replacement = cls(child.num_features, eps=child.eps,
                                  momentum=momentum, affine=child.affine)
                replacement.to(device=child.running_mean.device, dtype=child.running_mean.dtype)
                replacement.load_state_dict(child.state_dict(), strict=True)
                replacement.init_source_stats()
                replacement.train(child.training)
                setattr(parent, name, replacement)
            else:
                convert(child)
    convert(model)
    model.domain_bn = True


def set_bn_domain(model, domain):
    for module in model.modules():
        if isinstance(module, DomainStatsMixin):
            module.domain = domain


@torch.no_grad()
def refresh_target_bn(model, weak_images):
    """Update target statistics on weak images, then lock them for strong views."""
    model.eval()  # No dropout in the statistics pass.
    for module in model.modules():
        if isinstance(module, DomainStatsMixin):
            module.freeze_target_stats = False
            module.train()
    model(weak_images, None, None, mode="test", task="target")
    for module in model.modules():
        if isinstance(module, DomainStatsMixin):
            module.freeze_target_stats = True
    model.eval()


def inference_state_dict(model):
    """Keep original CAST parameter names; exported inference uses target BN."""
    return {key: value for key, value in model.state_dict().items()
            if key.rsplit(".", 1)[-1] not in {
                "source_running_mean", "source_running_var", "source_num_batches_tracked"}}
