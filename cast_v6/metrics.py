from __future__ import annotations

import json
import math
import os
from typing import Dict, List

import torch
import torch.nn.functional as F


class JsonlLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, record: Dict[str, object]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fmt(values, ndigits=4):
    return [round(float(v), ndigits) for v in values]


def print_pseudo_log(epoch: int, bank) -> None:
    """Print the compact pseudo-label summary.

    v6.2 renamed the old, ambiguous ``pseudo_class_accuracy`` diagnostic to
    ``pseudo_precision`` and ``pseudo_recall``.  Precision/recall are printed by
    train_v6.py's dedicated PseudoQuality line, so this compact line only adds
    overall pseudo accuracy and must not depend on the removed v6.1 attribute.
    """
    extra = ""
    if bank.pseudo_accuracy is not None:
        extra = " pseudo_acc=%.4f" % bank.pseudo_accuracy
    print(
        "[Epoch %d][Pseudo] agreement=%.4f thresholds=%s predicted=%s selected=%s "
        "selected_total=%d ratio=%.4f%s"
        % (
            epoch,
            bank.agreement_rate,
            _fmt(bank.thresholds),
            bank.predicted_counts,
            bank.selected_counts,
            int(bank.selected.sum().item()),
            bank.selected_ratio,
            extra,
        )
    )


def prediction_health(predicted_counts: List[int], selected_counts: List[int],
                      total: int) -> Dict[str, object]:
    p = torch.tensor(predicted_counts, dtype=torch.float32)
    s = torch.tensor(selected_counts, dtype=torch.float32)
    pred_dist = p / p.sum().clamp_min(1.0)
    sel_dist = s / s.sum().clamp_min(1.0)
    num_classes = len(predicted_counts)
    pred_entropy = float((-(pred_dist.clamp_min(1e-8) * pred_dist.clamp_min(1e-8).log()).sum()
                          / math.log(num_classes)).item())
    selected_classes = int((s > 0).sum().item())
    max_pred_ratio = float(pred_dist.max().item())
    min_pred_ratio = float(pred_dist.min().item())
    min_selected_ratio = float(sel_dist.min().item()) if float(s.sum().item()) > 0 else 0.0
    selected_ratio = float(s.sum().item() / max(1, total))
    reasons = []
    if max_pred_ratio > 0.55:
        reasons.append("prediction_collapse")
    if selected_classes < 5:
        reasons.append("low_class_coverage")
    if selected_ratio < 0.02:
        reasons.append("pseudo_starvation")
    # A class below 0.5% of all predictions is effectively starved even when it
    # is technically non-zero. This catches the FER fear failure seen in v6.
    if min_pred_ratio < 0.005:
        reasons.append("minority_prediction_starvation")
    if selected_classes == num_classes and min_selected_ratio < 0.002:
        reasons.append("minority_pseudo_starvation")
    return {
        "max_pred_ratio": max_pred_ratio,
        "min_pred_ratio": min_pred_ratio,
        "pred_entropy": pred_entropy,
        "selected_classes": selected_classes,
        "selected_ratio": selected_ratio,
        "min_selected_ratio": min_selected_ratio,
        "status": "WARN:" + ",".join(reasons) if reasons else "OK",
    }


@torch.no_grad()
def evaluate(model, loader, device: torch.device, num_classes: int = 7) -> Dict[str, object]:
    model.eval()
    correct = 0
    total = 0
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)
    predicted = torch.zeros(num_classes, dtype=torch.long)
    loss_sum = 0.0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        for c in range(num_classes):
            m = y == c
            class_total[c] += int(m.sum().item())
            class_correct[c] += int(((pred == y) & m).sum().item())
            predicted[c] += int((pred == c).sum().item())
    class_acc = (class_correct.float() / class_total.clamp_min(1).float()).tolist()
    return {
        "acc": correct / max(1, total),
        "loss": loss_sum / max(1, total),
        "class_acc": class_acc,
        "class_total": class_total.tolist(),
        "predicted": predicted.tolist(),
    }
