from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class PseudoBank:
    labels: torch.Tensor
    weights: torch.Tensor
    selected: torch.Tensor
    confidence: torch.Tensor
    agreement: torch.Tensor
    thresholds: torch.Tensor
    predicted_counts: List[int]
    selected_counts: List[int]
    agreement_rate: float
    selected_ratio: float
    pseudo_accuracy: Optional[float]
    pseudo_class_accuracy: Optional[List[float]]


def _normalized_entropy(prob: torch.Tensor) -> torch.Tensor:
    c = prob.size(1)
    ent = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(dim=1)
    return ent / torch.log(torch.tensor(float(c), device=prob.device))


@torch.no_grad()
def build_pseudo_bank(
    teacher,
    loader,
    dataset_size: int,
    num_classes: int,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    phi: float = 1.4,
    threshold_cap: float = 0.9,
    temperature: float = 1.0,
    min_margin: float = 0.0,
    max_entropy: float = 1.0,
    confidence_floor: float = 0.0,
    debug_target_labels: bool = False,
) -> PseudoBank:
    teacher.eval()
    probs = torch.zeros(dataset_size, num_classes, dtype=torch.float32)
    conf = torch.zeros(dataset_size, dtype=torch.float32)
    pred = torch.zeros(dataset_size, dtype=torch.long)
    agree = torch.zeros(dataset_size, dtype=torch.bool)
    margin = torch.zeros(dataset_size, dtype=torch.float32)
    entropy = torch.ones(dataset_size, dtype=torch.float32)
    gt = torch.full((dataset_size,), -1, dtype=torch.long)

    temperature = max(float(temperature), 1e-4)

    for batch in loader:
        w1, w2, labels, indices = batch
        w1 = w1.to(device, non_blocking=True)
        w2 = w2.to(device, non_blocking=True)
        l1, _ = teacher(w1)
        l2, _ = teacher(w2)
        p1 = F.softmax(l1 / temperature, dim=1)
        p2 = F.softmax(l2 / temperature, dim=1)
        p = (p1 + p2) * 0.5
        c, y = p.max(dim=1)
        y1 = p1.argmax(dim=1)
        y2 = p2.argmax(dim=1)
        top2 = p.topk(k=2, dim=1).values
        m = top2[:, 0] - top2[:, 1]
        e = _normalized_entropy(p)

        idx = indices.long()
        probs[idx] = p.cpu()
        conf[idx] = c.cpu()
        pred[idx] = y.cpu()
        agree[idx] = (y1 == y2).cpu()
        margin[idx] = m.cpu()
        entropy[idx] = e.cpu()
        if debug_target_labels:
            gt[idx] = labels.long().cpu()

    thresholds = torch.full((num_classes,), float(threshold_cap), dtype=torch.float32)
    pred_counts = []
    stage_factor = float(total_epochs) / float(max(1, total_epochs - epoch))
    for c in range(num_classes):
        mask = pred == c
        count = int(mask.sum().item())
        pred_counts.append(count)
        if count > 0:
            pc = float(conf[mask].mean().item()) * float(phi)
            thresholds[c] = min(pc * stage_factor, float(threshold_cap))

    sample_thr = thresholds[pred]
    selected = (
        agree
        & (conf >= sample_thr)
        & (conf >= float(confidence_floor))
        & (margin >= float(min_margin))
        & (entropy <= float(max_entropy))
    )

    weights = conf.clamp(0.05, 1.0)
    weights = weights * selected.float()

    selected_counts = [int(((pred == c) & selected).sum().item()) for c in range(num_classes)]
    selected_total = int(selected.sum().item())

    pseudo_acc = None
    pseudo_class_acc = None
    if debug_target_labels and selected_total > 0:
        pseudo_acc = float((pred[selected] == gt[selected]).float().mean().item())
        pseudo_class_acc = []
        for c in range(num_classes):
            m = selected & (pred == c)
            if int(m.sum().item()) == 0:
                pseudo_class_acc.append(0.0)
            else:
                pseudo_class_acc.append(float((pred[m] == gt[m]).float().mean().item()))

    return PseudoBank(
        labels=pred,
        weights=weights,
        selected=selected,
        confidence=conf,
        agreement=agree,
        thresholds=thresholds,
        predicted_counts=pred_counts,
        selected_counts=selected_counts,
        agreement_rate=float(agree.float().mean().item()),
        selected_ratio=float(selected.float().mean().item()),
        pseudo_accuracy=pseudo_acc,
        pseudo_class_accuracy=pseudo_class_acc,
    )


def pseudo_bank_summary(bank: PseudoBank) -> Dict[str, object]:
    return {
        "thresholds": [round(float(x), 4) for x in bank.thresholds],
        "predicted": bank.predicted_counts,
        "selected": bank.selected_counts,
        "agreement": round(bank.agreement_rate, 4),
        "selected_ratio": round(bank.selected_ratio, 4),
        "pseudo_acc": None if bank.pseudo_accuracy is None else round(bank.pseudo_accuracy, 4),
        "pseudo_class_acc": None if bank.pseudo_class_accuracy is None else [
            round(float(x), 4) for x in bank.pseudo_class_accuracy
        ],
    }
