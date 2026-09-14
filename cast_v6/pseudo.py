from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

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
    pseudo_precision: Optional[List[float]]
    pseudo_recall: Optional[List[float]]
    pseudo_confusion: Optional[List[List[int]]]
    raw_prior: List[float]
    correction: List[float]
    keep_ratio: float
    class_weights: List[float]


def _normalized_entropy(prob: torch.Tensor) -> torch.Tensor:
    c = prob.size(1)
    ent = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(dim=1)
    return ent / torch.log(torch.tensor(float(c), device=prob.device))


def _quantile_value(values: torch.Tensor, q: float) -> float:
    """Version-safe scalar quantile without depending on torch.quantile."""
    if values.numel() == 0:
        return 1.0
    values = torch.sort(values.flatten()).values
    q = min(max(float(q), 0.0), 1.0)
    idx = int(round(q * float(values.numel() - 1)))
    return float(values[idx].item())


def _apply_class_correction(prob: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
    adjusted = prob * correction.view(1, -1)
    return adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(1e-8)


def source_class_correction(
    class_total: Sequence[int],
    predicted_counts: Sequence[int],
    alpha: float = 0.5,
    min_correction: float = 0.85,
    max_correction: float = 1.5,
) -> List[float]:
    """Calibrate class bias using labeled *source* data only.

    CAST's RAF checkpoint under-predicts some source classes (notably fear and
    angry).  The previous v6.1 target-side uniform distribution alignment was
    unsuitable for FER2013 because FER2013 is strongly imbalanced (e.g. disgust
    is genuinely rare).  This correction instead compares source ground-truth
    class frequency with the source model's prediction frequency, so it does
    not assume any target class prior and does not consume target labels.
    """
    true = torch.tensor(list(class_total), dtype=torch.float32).clamp_min(1.0)
    pred = torch.tensor(list(predicted_counts), dtype=torch.float32).clamp_min(1.0)
    if true.numel() != pred.numel():
        raise ValueError("class_total and predicted_counts must have the same length")
    ratio = (true / true.sum()) / (pred / pred.sum()).clamp_min(1e-8)
    corr = ratio.pow(max(0.0, float(alpha)))
    corr = corr.clamp(float(min_correction), float(max_correction))
    return [float(x) for x in corr.tolist()]


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
    threshold_cap: float = 0.95,
    temperature: float = 1.0,
    min_margin: float = 0.0,
    max_entropy: float = 1.0,
    confidence_floor: float = 0.0,
    keep_ratio_start: float = 0.35,
    keep_ratio_end: float = 0.55,
    class_correction: Optional[Sequence[float]] = None,
    class_balance_max: float = 1.25,
    class_weight_correction_gate: float = 1.10,
    debug_target_labels: bool = False,
) -> PseudoBank:
    """Generate one whole-target pseudo bank from two EMA weak views.

    Target labels are only read when ``debug_target_labels`` is enabled and are
    never used in pseudo-label selection, correction, weighting, or loss.

    ``class_correction`` must be derived without target labels.  v6.2 supplies a
    conservative correction estimated from labeled RAF source data.  This
    replaces v6.1's uniform target-prior assumption, which incorrectly boosted
    FER2013 disgust because the canonical FER2013 distribution is not uniform.
    """
    teacher.eval()
    p1_all = torch.zeros(dataset_size, num_classes, dtype=torch.float32)
    p2_all = torch.zeros(dataset_size, num_classes, dtype=torch.float32)
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

        idx = indices.long()
        p1_all[idx] = p1.cpu()
        p2_all[idx] = p2.cpu()
        if debug_target_labels:
            gt[idx] = labels.long().cpu()

    raw_prob = (p1_all + p2_all) * 0.5
    raw_prior_t = raw_prob.mean(dim=0).clamp_min(1e-8)

    if class_correction is None:
        correction = torch.ones(num_classes, dtype=torch.float32)
    else:
        correction = torch.tensor(list(class_correction), dtype=torch.float32)
        if correction.numel() != num_classes:
            raise ValueError("class_correction must have num_classes values")
        correction = correction.clamp_min(1e-4)

    p1_adj = _apply_class_correction(p1_all, correction)
    p2_adj = _apply_class_correction(p2_all, correction)
    prob = (p1_adj + p2_adj) * 0.5

    conf, pred = prob.max(dim=1)
    y1 = p1_adj.argmax(dim=1)
    y2 = p2_adj.argmax(dim=1)
    agree = y1 == y2
    top2 = prob.topk(k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    entropy = _normalized_entropy(prob)

    progress = float(epoch) / float(max(1, total_epochs - 1))
    keep_ratio = float(keep_ratio_start) + progress * (float(keep_ratio_end) - float(keep_ratio_start))
    keep_ratio = min(max(keep_ratio, 0.05), 0.95)

    quality_mask = (
        agree
        & (conf >= float(confidence_floor))
        & (margin >= float(min_margin))
        & (entropy <= float(max_entropy))
    )

    thresholds = torch.full((num_classes,), float(threshold_cap), dtype=torch.float32)
    pred_counts: List[int] = []

    for c in range(num_classes):
        class_mask = pred == c
        count = int(class_mask.sum().item())
        pred_counts.append(count)
        if count == 0:
            continue

        candidate_conf = conf[class_mask & quality_mask]
        if candidate_conf.numel() == 0:
            continue

        quantile_thr = _quantile_value(candidate_conf, 1.0 - keep_ratio)
        class_mean = float(conf[class_mask].mean().item())
        catm_ceiling = min(class_mean * float(phi), float(threshold_cap))
        threshold = min(quantile_thr, catm_ceiling, float(threshold_cap))
        threshold = max(threshold, float(confidence_floor))
        thresholds[c] = threshold

    sample_thr = thresholds[pred]
    selected = quality_mask & (conf >= sample_thr)

    selected_counts = [int(((pred == c) & selected).sum().item()) for c in range(num_classes)]
    selected_total = int(selected.sum().item())

    # Only source-calibrated under-predicted classes may receive a small pseudo
    # CE boost.  This prevents a low-count but unreliable target pseudo class
    # (the v6.1 disgust failure) from being amplified merely because it is rare.
    selected_count_t = torch.tensor(selected_counts, dtype=torch.float32)
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    nonzero = selected_count_t > 0
    if bool(nonzero.any()) and float(class_balance_max) > 1.0:
        mean_nonzero = selected_count_t[nonzero].mean().clamp_min(1.0)
        candidate = torch.sqrt(mean_nonzero / selected_count_t.clamp_min(1.0))
        candidate = candidate.clamp(1.0, max(1.0, float(class_balance_max)))
        eligible = correction >= float(class_weight_correction_gate)
        class_weights[eligible & nonzero] = candidate[eligible & nonzero]

    weights = conf.clamp(0.05, 1.0) * class_weights[pred]
    weights = weights * selected.float()

    pseudo_acc = None
    pseudo_precision = None
    pseudo_recall = None
    pseudo_confusion = None
    if debug_target_labels and selected_total > 0:
        correct = pred[selected] == gt[selected]
        pseudo_acc = float(correct.float().mean().item())

        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        true_sel = gt[selected]
        pred_sel = pred[selected]
        flat = true_sel * num_classes + pred_sel
        bincount = torch.bincount(flat, minlength=num_classes * num_classes)
        confusion = bincount.view(num_classes, num_classes)

        diag = confusion.diag().float()
        predicted_selected = confusion.sum(dim=0).float().clamp_min(1.0)
        true_total = torch.bincount(gt.clamp_min(0), minlength=num_classes).float().clamp_min(1.0)
        pseudo_precision = [float(x) for x in (diag / predicted_selected).tolist()]
        pseudo_recall = [float(x) for x in (diag / true_total).tolist()]
        pseudo_confusion = [[int(v) for v in row] for row in confusion.tolist()]

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
        pseudo_precision=pseudo_precision,
        pseudo_recall=pseudo_recall,
        pseudo_confusion=pseudo_confusion,
        raw_prior=[float(x) for x in raw_prior_t.tolist()],
        correction=[float(x) for x in correction.tolist()],
        keep_ratio=keep_ratio,
        class_weights=[float(x) for x in class_weights.tolist()],
    )


def pseudo_bank_summary(bank: PseudoBank) -> Dict[str, object]:
    return {
        "thresholds": [round(float(x), 4) for x in bank.thresholds],
        "predicted": bank.predicted_counts,
        "selected": bank.selected_counts,
        "agreement": round(bank.agreement_rate, 4),
        "selected_ratio": round(bank.selected_ratio, 4),
        "keep_ratio": round(bank.keep_ratio, 4),
        "raw_prior": [round(float(x), 4) for x in bank.raw_prior],
        "correction": [round(float(x), 4) for x in bank.correction],
        "class_weights": [round(float(x), 4) for x in bank.class_weights],
        "pseudo_acc": None if bank.pseudo_accuracy is None else round(bank.pseudo_accuracy, 4),
        "pseudo_precision": None if bank.pseudo_precision is None else [round(float(x), 4) for x in bank.pseudo_precision],
        "pseudo_recall": None if bank.pseudo_recall is None else [round(float(x), 4) for x in bank.pseudo_recall],
        "pseudo_confusion": bank.pseudo_confusion,
    }
