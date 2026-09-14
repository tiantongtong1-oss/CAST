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


def _apply_distribution_alignment(prob: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
    adjusted = prob * correction.view(1, -1)
    return adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(1e-8)


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
    distribution_align_alpha: float = 0.35,
    distribution_align_max: float = 2.0,
    class_balance_max: float = 2.0,
    debug_target_labels: bool = False,
) -> PseudoBank:
    """Generate one whole-target pseudo bank from two EMA weak views.

    Compared with the first v6 implementation, this version addresses the
    observed confirmation-bias pattern (agreement rising while pseudo accuracy
    falls):
      1. conservative distribution alignment before argmax;
      2. class-wise quantile thresholds instead of thresholds saturating at one
         global 0.9 cap;
      3. a scheduled pseudo-label budget so coverage cannot grow unchecked;
      4. inverse-sqrt class weights, clipped to avoid amplifying noisy minority
         pseudo labels excessively.

    Target labels are only read when debug_target_labels=True and never affect
    selection, weighting, or optimization.
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
    raw_prior_t = raw_prob.mean(dim=0).clamp_min(1e-6)

    # Unsupervised distribution alignment. Uniform is deliberately used rather
    # than target labels; the correction is weak and clipped so it cannot force
    # an artificial uniform prediction histogram in one step.
    uniform = torch.full_like(raw_prior_t, 1.0 / float(num_classes))
    alpha = max(0.0, float(distribution_align_alpha))
    max_corr = max(1.0, float(distribution_align_max))
    correction = (uniform / raw_prior_t).pow(alpha)
    correction = correction.clamp(1.0 / max_corr, max_corr)

    p1_adj = _apply_distribution_alignment(p1_all, correction)
    p2_adj = _apply_distribution_alignment(p2_all, correction)
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

        # Keep roughly a class-wise fraction of the reliable pool. The paper's
        # CATM mean-confidence term is retained as a ceiling, but no longer
        # allowed to force every class to the same saturated threshold.
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

    # Confidence weighting + clipped inverse-sqrt pseudo-class weighting.
    # This compensates class starvation without letting a tiny/noisy class
    # dominate the target loss.
    selected_count_t = torch.tensor(selected_counts, dtype=torch.float32)
    nonzero = selected_count_t > 0
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    if bool(nonzero.any()):
        mean_nonzero = selected_count_t[nonzero].mean().clamp_min(1.0)
        class_weights[nonzero] = torch.sqrt(mean_nonzero / selected_count_t[nonzero].clamp_min(1.0))
        class_weights = class_weights.clamp(1.0 / max(1.0, float(class_balance_max)),
                                            max(1.0, float(class_balance_max)))

    weights = conf.clamp(0.05, 1.0) * class_weights[pred]
    weights = weights * selected.float()

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
        "pseudo_class_acc": None if bank.pseudo_class_accuracy is None else [
            round(float(x), 4) for x in bank.pseudo_class_accuracy
        ],
    }
