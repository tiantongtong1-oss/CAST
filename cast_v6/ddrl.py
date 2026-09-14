from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _multi_rbf_kernel(x: torch.Tensor, y: torch.Tensor,
                      kernel_mul: float = 2.0,
                      kernel_num: int = 5) -> torch.Tensor:
    z = torch.cat([x, y], dim=0)
    d2 = torch.cdist(z, z, p=2).pow(2)
    nonzero = d2.detach()[d2.detach() > 0]
    base = nonzero.median() if nonzero.numel() else torch.tensor(1.0, device=z.device)
    base = base.clamp_min(1e-4) / (kernel_mul ** (kernel_num // 2))
    kernels = 0.0
    for i in range(kernel_num):
        bandwidth = base * (kernel_mul ** i)
        kernels = kernels + torch.exp(-d2 / bandwidth.clamp_min(1e-6))
    # Average instead of summing kernels. This preserves MK-MMD geometry while
    # keeping the scale independent of kernel_num.
    return kernels / float(max(1, kernel_num))


def mmd2(x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
    """Unbiased MK-MMD^2. Returns None when either set has <2 samples."""
    if x.size(0) < 2 or y.size(0) < 2:
        return None
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    n, m = x.size(0), y.size(0)
    k = _multi_rbf_kernel(x, y)
    kxx = k[:n, :n]
    kyy = k[n:, n:]
    kxy = k[:n, n:]
    sum_xx = (kxx.sum() - kxx.diag().sum()) / float(n * (n - 1))
    sum_yy = (kyy.sum() - kyy.diag().sum()) / float(m * (m - 1))
    sum_xy = kxy.mean()
    return sum_xx + sum_yy - 2.0 * sum_xy


def ddrl_loss(source_features: torch.Tensor,
              source_labels: torch.Tensor,
              target_features: torch.Tensor,
              target_labels: torch.Tensor,
              eta: torch.Tensor,
              num_classes: int = 7,
              min_class_samples: int = 2,
              max_abs_loss: float = 1.5) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Class-conditional MK-MMD with missing-class-safe normalization.

    `loss = intra - inter`, matching CAST's align-within/separate-between idea.
    The final value is clipped only as a safety bound; with averaged kernels it
    should normally stay well inside the interval.
    """
    intra_terms, inter_terms = [], []
    intra_classes, inter_classes = [], []

    all_features = torch.cat([source_features, target_features], dim=0)
    all_labels = torch.cat([source_labels, target_labels], dim=0)

    for c in range(num_classes):
        xs = source_features[source_labels == c]
        xt = target_features[target_labels == c]
        if xs.size(0) >= min_class_samples and xt.size(0) >= min_class_samples:
            term = mmd2(xs, xt)
            if term is not None and torch.isfinite(term):
                intra_terms.append(term * eta[c])
                intra_classes.append(c)

        xc = all_features[all_labels == c]
        xr = all_features[all_labels != c]
        if xc.size(0) >= min_class_samples and xr.size(0) >= min_class_samples:
            term = mmd2(xc, xr)
            if term is not None and torch.isfinite(term):
                inter_terms.append(term * eta[c])
                inter_classes.append(c)

    zero = source_features.sum() * 0.0
    intra = torch.stack(intra_terms).mean() if intra_terms else zero
    inter = torch.stack(inter_terms).mean() if inter_terms else zero
    raw_loss = intra - inter
    loss = raw_loss.clamp(min=-float(max_abs_loss), max=float(max_abs_loss))
    info = {
        "intra": float(intra.detach().item()),
        "inter": float(inter.detach().item()),
        "raw_loss": float(raw_loss.detach().item()),
        "loss": float(loss.detach().item()),
        "intra_classes": intra_classes,
        "inter_classes": inter_classes,
        "active_classes": len(intra_classes),
    }
    return loss, info
