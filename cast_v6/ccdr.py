from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def class_volume_weights(features: torch.Tensor, labels: torch.Tensor,
                         num_classes: int = 7,
                         eta_min: float = 0.25,
                         eta_max: float = 4.0) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """CCDR class-level representation modulation.

    Implements the paper's density -> effective volume -> eta=1/V idea using
    a numerically stable torch RBF density estimate. Features are L2-normalized
    before density estimation, and eta is normalized to mean 1 over valid classes
    so it does not silently rescale the full objective.
    """
    device = features.device
    feat = F.normalize(features.detach(), dim=1)
    volumes = torch.full((num_classes,), float("nan"), device=device)
    eta = torch.ones(num_classes, device=device)
    valid = torch.zeros(num_classes, dtype=torch.bool, device=device)

    for c in range(num_classes):
        x = feat[labels == c]
        n = x.size(0)
        if n < 2:
            continue
        d2 = torch.cdist(x, x, p=2).pow(2)
        nonzero = d2[d2 > 0]
        if nonzero.numel() == 0:
            bandwidth2 = torch.tensor(1.0, device=device)
        else:
            bandwidth2 = nonzero.median().clamp_min(1e-4)
        kernel = torch.exp(-d2 / (2.0 * bandwidth2))
        rho = kernel.mean(dim=1).clamp_min(1e-6)
        volume = (1.0 / rho).sum()
        volumes[c] = volume
        eta[c] = 1.0 / volume.clamp_min(1e-6)
        valid[c] = True

    if valid.any():
        mean_eta = eta[valid].mean().clamp_min(1e-6)
        eta[valid] = (eta[valid] / mean_eta).clamp(eta_min, eta_max)
        eta[~valid] = 1.0

    info = {
        "volume": [None if torch.isnan(v) else float(v.item()) for v in volumes],
        "eta": [float(v.item()) for v in eta],
        "valid": [bool(v.item()) for v in valid],
    }
    return eta, info


def classifier_modulation_loss(weight: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Paper Eq. 6 without the constant diagonal terms."""
    w = F.normalize(weight, dim=1)
    sim = w @ w.t()
    c = sim.size(0)
    offdiag = ~torch.eye(c, dtype=torch.bool, device=sim.device)
    values = sim[offdiag]
    loss = ((values + 1.0) * 0.5).mean()
    return loss, float(values.detach().mean().item())
