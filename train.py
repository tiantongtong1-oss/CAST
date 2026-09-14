import argparse
import copy
import json
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

import Networks
import image_utils as util
from dataset import FER, RafDataSet
from domain_bn import enable_domain_bn, inference_state_dict, refresh_target_bn
from randaugment import RandAugmentMC

warnings.filterwarnings("ignore")

SEED = 1314
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class IndexedTargetDataset(Dataset):
    """Wrap FER training data with a stable sample id for temporal pseudo labels."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        weak1, weak2, strong, label = self.base[idx]
        return weak1, weak2, strong, label, idx


class TargetScanDataset(Dataset):
    """Use a deterministic original/flip pair for epoch-level pseudo labels."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        return image, torch.flip(image, dims=[-1]), image, label, idx


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, batch_size, workers, shuffle, drop_last, seed_offset):
    generator = torch.Generator()
    generator.manual_seed(SEED + seed_offset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_transforms(augmentation="face"):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    source = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomCrop(224, padding=32),
        ], p=0.5),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
    weak = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    strong = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomCrop(224, padding=32),
        ], p=0.5),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
    test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    if augmentation == "face":
        # FER2013 faces are only 48 x 48. Preserve the eyes/mouth and avoid
        # stacking severe crops, solarization, rotation and large erasing.
        source = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=12, padding_mode="reflect"),
            transforms.RandomRotation(10),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
        ])
        strong = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=12, padding_mode="reflect"),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(), normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
        ])
    return {"source": source, "weak": weak, "strong": strong, "test": test}


def classifier_weight_loss(model):
    weight = F.normalize(model.fc.weight, dim=1)
    identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
    return ((weight.mm(weight.t()) - identity + 1.0) / 2.0).mean()


def freeze_backbone_bn_stats(model):
    """Freeze all BN statistics, including the seven-logit classification head.

    BN affine parameters still receive gradients. Mixed source/strong-target
    batches must not change the statistics used by the weak-view EMA teacher.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def optimizer_groups(model, head_lr, backbone_multiplier):
    """Fine-tune pretrained convolutions more slowly than the new projection."""
    projection = [p for layer in model.feature.children()
                  if isinstance(layer, nn.Linear) for p in layer.parameters()]
    projection_ids = {id(p) for p in projection}
    backbone = [p for p in model.feature.parameters() if id(p) not in projection_ids]
    head = projection + list(model.fc.parameters()) + list(model.bn.parameters())
    return [{"params": backbone, "lr": head_lr * backbone_multiplier},
            {"params": head, "lr": head_lr}]


@torch.no_grad()
def update_ema(student, teacher, decay):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        # Running moments already use EMA. Smoothing them again with 0.999
        # delays normalization by thousands of batches after feature updates.
        teacher_buffer.data.copy_(student_buffer.data)


@torch.no_grad()
def evaluate(model, loader, num, task="target", verbose=True):
    model.eval()
    correct = 0
    preds, labels = [], []
    for imgs, targets in loader:
        imgs = imgs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        logits, _ = model(imgs, targets, None, mode="test", task=task)
        pred = logits.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        preds.append(pred.cpu())
        labels.append(targets.cpu())
    if verbose:
        util.make_confucion_matrix(preds, labels)
    return float(np.around(correct / float(max(num, 1)), 4))


def save_checkpoint(model, optimizer, path, epoch, accuracy, kind):
    torch.save(
        {
            "model": inference_state_dict(model),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "accuracy": accuracy,
            "model_kind": kind,
            "training_domain_bn": getattr(model, "domain_bn", False),
        },
        path,
    )


@torch.no_grad()
def adapt_backbone_bn(model, target_loader, max_batches, momentum):
    """Recalibrate backbone, then head, using only clean target-train images."""
    if max_batches <= 0:
        return

    model.eval()
    bn_layers = []
    old_momentum = []
    for module in model.feature.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            bn_layers.append(module)
            old_momentum.append(module.momentum)
            module.momentum = momentum
            module.train()

    used = 0
    for weak1, _, _, _, _ in target_loader:
        model.feature(weak1.cuda(non_blocking=True))
        used += 1
        if used >= max_batches:
            break

    for module, old in zip(bn_layers, old_momentum):
        module.momentum = old
    model.eval()

    # Calibrate the head with the *final* backbone statistics and no dropout.
    # Otherwise a calibrated backbone feeds a head normalized for source data.
    old_head_momentum = model.bn.momentum
    model.bn.momentum = momentum
    model.bn.train()
    head_used = 0
    for weak1, _, _, _, _ in target_loader:
        if weak1.shape[0] < 2:
            continue
        features = model.feature(weak1.cuda(non_blocking=True))
        model.bn(model.fc(features))
        head_used += 1
        if head_used >= max_batches:
            break
    model.bn.momentum = old_head_momentum
    model.eval()
    print("Target BN recalibration batches: backbone=%d head=%d" % (used, head_used))


@torch.no_grad()
def build_source_prototypes(anchor, loader, class_num=7, centers_per_class=3):
    anchor.eval()
    all_features, all_targets = [], []
    counts = torch.zeros(class_num, dtype=torch.long)

    for imgs, targets in loader:
        imgs = imgs.cuda(non_blocking=True)
        _, features = anchor(imgs, None, None, mode="test", task="source")
        features = F.normalize(features.float(), dim=1)
        targets = targets.long().cpu()

        all_features.append(features)
        all_targets.append(targets)
        for c in range(class_num):
            mask = targets.eq(c)
            if mask.any():
                counts[c] += int(mask.sum().item())

    if not all_features or (counts == 0).any():
        raise RuntimeError("Incomplete source prototypes: %s" % counts.tolist())
    features = torch.cat(all_features)
    targets = torch.cat(all_targets)
    center = features.mean(dim=0, keepdim=True)
    features = F.normalize(features - center, dim=1)
    prototypes = []
    for c in range(class_num):
        class_features = features[targets == c]
        # Deterministic spherical clustering covers multiple expression modes;
        # centering removes common face components from cosine comparisons.
        centers = [F.normalize(class_features.mean(0), dim=0)]
        for _ in range(1, centers_per_class):
            nearest = class_features.mm(torch.stack(centers).t()).max(1).values
            centers.append(class_features[nearest.argmin()])
        centers = torch.stack(centers)
        for _ in range(10):
            assignment = class_features.mm(centers.t()).argmax(1)
            for k in range(centers_per_class):
                members = class_features[assignment == k]
                if len(members):
                    centers[k] = F.normalize(members.mean(0), dim=0)
        prototypes.append(centers)
    return {"center": center, "centers": torch.stack(prototypes)}, counts


def prototype_similarity(features, bank):
    centered = F.normalize(F.normalize(features.float(), dim=1) - bank["center"], dim=1)
    centers = bank["centers"]
    return centered.mm(centers.flatten(0, 1).t()).view(len(features), len(centers), -1).max(2).values


def smoothed_cross_entropy(logits, targets, smoothing):
    log_prob = F.log_softmax(logits, dim=1)
    nll = -log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
    return ((1.0 - smoothing) * nll - smoothing * log_prob.mean(dim=1)).mean()


@torch.no_grad()
def calibrate_temperature(model, source_val_loader):
    """Select a scalar temperature by source-validation NLL, never target labels."""
    model.eval()
    logits, labels = [], []
    for images, targets in source_val_loader:
        out, _ = model(images.cuda(non_blocking=True), None, None, "test", "source")
        logits.append(out.cpu())
        labels.append(targets.long().cpu())
    logits, labels = torch.cat(logits), torch.cat(labels)
    temperatures = torch.logspace(0.0, np.log10(8.0), steps=41)
    losses = torch.stack([F.cross_entropy(logits / t, labels) for t in temperatures])
    temperature = float(temperatures[losses.argmin()])
    print("Source-validation temperature %.4f NLL %.4f -> %.4f"
          % (temperature, float(losses[0]), float(losses.min())))
    return temperature


def tempered_probability(logits, temperature):
    return F.softmax(logits / float(max(temperature, 1e-6)), dim=1)


@torch.no_grad()
def estimate_target_thresholds(teacher, anchor, loader, args, epoch, previous=None):
    teacher.eval()
    anchor.eval()
    class_all = [[] for _ in range(7)]
    class_anchor = [[] for _ in range(7)]
    class_stable = [[] for _ in range(7)]
    predicted = torch.zeros(7, dtype=torch.long)
    total = 0
    weak_agree = 0
    anchor_agree = 0

    for weak1, weak2, _, _, _ in loader:
        weak1 = weak1.cuda(non_blocking=True)
        weak2 = weak2.cuda(non_blocking=True)
        out1, _ = teacher(weak1, None, None, "test", "target")
        out2, _ = teacher(weak2, None, None, "test", "target")
        anchor_out, _ = anchor(weak1, None, None, "test", "target")

        prob1 = tempered_probability(out1, args.teacher_temperature).cpu()
        prob2 = tempered_probability(out2, args.teacher_temperature).cpu()
        anchor_prob = tempered_probability(anchor_out, args.teacher_temperature).cpu()
        avg = 0.5 * (prob1 + prob2)
        confidence, target = avg.max(dim=1)
        agreement = prob1.argmax(dim=1).eq(prob2.argmax(dim=1))
        anchor_conf, anchor_pred = anchor_prob.max(dim=1)
        anchor_match = anchor_pred.eq(target) & (anchor_conf >= args.anchor_min_confidence)
        stable_conf = torch.minimum(
            prob1.gather(1, target.unsqueeze(1)).squeeze(1),
            prob2.gather(1, target.unsqueeze(1)).squeeze(1),
        )

        total += target.numel()
        weak_agree += int(agreement.sum().item())
        anchor_agree += int((agreement & anchor_match).sum().item())
        predicted += torch.bincount(target, minlength=7)

        for c in range(7):
            all_values = confidence[target == c]
            if all_values.numel():
                class_all[c].append(all_values)
            consistent_values = stable_conf[(target == c) & agreement]
            if consistent_values.numel():
                class_stable[c].append(consistent_values)
            stable_values = stable_conf[(target == c) & agreement & anchor_match]
            if stable_values.numel():
                class_anchor[c].append(stable_values)

    progress = float(epoch) / float(max(args.epochs - 1, 1))
    quantile = min(args.threshold_quantile + 0.05 * progress, 0.90)
    stage_factor = float(args.epochs) / float(max(args.epochs - epoch, 1))
    thresholds = torch.full((7,), args.pseudo_max_threshold, dtype=torch.float32)
    support = torch.zeros(7, dtype=torch.long)

    for c in range(7):
        if not class_all[c]:
            continue
        values = torch.cat(class_all[c])
        cast_threshold = float(values.mean().item()) * args.phi * stage_factor
        cast_threshold = float(np.clip(
            cast_threshold, args.pseudo_min_threshold, args.pseudo_max_threshold
        ))
        threshold = cast_threshold
        if class_anchor[c]:
            stable_values = torch.cat(class_anchor[c])
            support[c] = stable_values.numel()
            if stable_values.numel() >= 8:
                q_threshold = float(torch.quantile(stable_values, quantile).item())
                threshold = 0.80 * cast_threshold + 0.20 * q_threshold
        if args.threshold_mode == "quantile":
            # Estimate the same min-view confidence used by the selector.
            # No epochs/(epochs-epoch) factor: it saturates all classes late
            # in training, defeating class adaptation and starving hard classes.
            if class_stable[c]:
                stable_values = torch.cat(class_stable[c])
                threshold = float(torch.quantile(stable_values, quantile).item())
            else:
                threshold = args.pseudo_max_threshold
            if previous is not None:
                threshold = (args.threshold_momentum * float(previous[c])
                             + (1.0 - args.threshold_momentum) * threshold)
        thresholds[c] = float(np.clip(
            threshold, args.pseudo_min_threshold, args.pseudo_max_threshold
        ))

    return (
        thresholds,
        support,
        predicted,
        weak_agree / float(max(total, 1)),
        anchor_agree / float(max(total, 1)),
    )


@torch.no_grad()
def build_epoch_pseudo_bank(
    teacher,
    anchor,
    loader,
    prototypes,
    thresholds,
    threshold_support,
    temporal_label,
    temporal_streak,
    args,
    epoch,
    previous_counts=None,
):
    """Build a balanced bank and seed missing classes without requiring teacher top-1."""
    teacher.eval()
    anchor.eval()
    n_samples = len(loader.dataset)

    top1_label = torch.full((n_samples,), -1, dtype=torch.long)
    trusted = torch.zeros(n_samples, dtype=torch.bool)
    trusted_score = torch.zeros(n_samples, dtype=torch.float32)
    avg_prob_all = torch.zeros((n_samples, 7), dtype=torch.float32)
    stable_prob_all = torch.zeros((n_samples, 7), dtype=torch.float32)
    anchor_prob_all = torch.zeros((n_samples, 7), dtype=torch.float32)
    proto_sim_all = torch.zeros((n_samples, 7), dtype=torch.float32)
    teacher_rank_all = torch.full((n_samples, 7), 7, dtype=torch.long)
    proto_pred_all = torch.full((n_samples,), -1, dtype=torch.long)
    proto_margin_all = torch.zeros(n_samples, dtype=torch.float32)

    strict_anchor = epoch < args.anchor_guard_epochs

    for weak1, weak2, _, _, sample_idx in loader:
        weak1 = weak1.cuda(non_blocking=True)
        weak2 = weak2.cuda(non_blocking=True)
        idx = sample_idx.long().cpu()

        out1, _ = teacher(weak1, None, None, "test", "target")
        out2, _ = teacher(weak2, None, None, "test", "target")
        anchor_out, anchor_features = anchor(weak1, None, None, "test", "target")

        prob1 = tempered_probability(out1, args.teacher_temperature)
        prob2 = tempered_probability(out2, args.teacher_temperature)
        avg = 0.5 * (prob1 + prob2)
        stable_all = torch.minimum(prob1, prob2)
        target = avg.argmax(dim=1)
        agreement = prob1.argmax(dim=1).eq(prob2.argmax(dim=1))
        stable_top1 = stable_all.gather(1, target.unsqueeze(1)).squeeze(1)
        sample_threshold = thresholds.to(target.device)[target]

        top2 = torch.topk(avg, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(avg.clamp_min(1e-8) * avg.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / np.log(7.0)

        anchor_prob = tempered_probability(anchor_out, args.teacher_temperature)
        anchor_conf, anchor_pred = anchor_prob.max(dim=1)
        anchor_match = anchor_pred.eq(target) & (anchor_conf >= args.anchor_min_confidence)

        similarity = prototype_similarity(anchor_features, prototypes)
        proto_top2 = torch.topk(similarity, k=2, dim=1)
        proto_pred = proto_top2.indices[:, 0]
        proto_margin = proto_top2.values[:, 0] - proto_top2.values[:, 1]
        proto_match = proto_pred.eq(target.cpu()) & (
            proto_margin >= args.prototype_min_margin
        )

        base = (
            agreement
            & (stable_top1 >= sample_threshold)
            & (margin >= args.min_margin)
            & (entropy <= args.max_entropy)
        ).cpu()
        anchor_cpu = anchor_match.cpu()
        if strict_anchor:
            trusted_batch = base & anchor_cpu & proto_match
        else:
            # The v4 OR gate can admit a confidently wrong class when
            # teacher and anchor shared a bias. Geometry remains mandatory.
            trusted_batch = base & proto_match

        proto_quality = ((proto_margin - args.prototype_min_margin) / 0.35).clamp(0.0, 1.0)
        quality = (
            0.45 * stable_top1.cpu()
            + 0.20 * margin.cpu()
            + 0.15 * (1.0 - entropy.cpu())
            + 0.10 * anchor_conf.cpu()
            + 0.10 * proto_quality
        )

        avg_cpu = avg.cpu()
        # rank[c] = 1 + count(P(other) > P(c)); highest probability is rank 1.
        rank = (avg_cpu.unsqueeze(1) > avg_cpu.unsqueeze(2)).sum(dim=2) + 1

        top1_label[idx] = target.cpu()
        trusted[idx] = trusted_batch
        trusted_score[idx] = quality
        avg_prob_all[idx] = avg_cpu
        stable_prob_all[idx] = stable_all.cpu()
        anchor_prob_all[idx] = anchor_prob.cpu()
        proto_sim_all[idx] = ((similarity.cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
        teacher_rank_all[idx] = rank.long()
        proto_pred_all[idx] = proto_pred
        proto_margin_all[idx] = proto_margin

    trusted_counts = torch.zeros(7, dtype=torch.long)
    for c in range(7):
        trusted_counts[c] = int((trusted & top1_label.eq(c)).sum().item())

    # Recover any class that has too few trusted samples. Recovery is class-conditioned:
    # the teacher does not need to predict class c as top-1. We rank samples using
    # P_teacher(c), weak-view stability, source-prototype similarity and anchor P(c).
    recovery_needed = (
        (trusted_counts < args.recovery_trigger_count)
        | (threshold_support < args.starvation_support)
    )
    recovery_score = (
        0.40 * stable_prob_all
        + 0.15 * avg_prob_all
        + 0.30 * proto_sim_all
        + 0.15 * anchor_prob_all
    )

    proposed_label = torch.full((n_samples,), -1, dtype=torch.long)
    proposed_score = torch.zeros(n_samples, dtype=torch.float32)
    proposed_recovery = torch.zeros(n_samples, dtype=torch.bool)
    proposed_label[trusted] = top1_label[trusted]
    proposed_score[trusted] = trusted_score[trusted]

    proposals = []
    for c in range(7):
        if not bool(recovery_needed[c]):
            continue
        # Absolute cosine similarity alone is insufficient: unrelated classes
        # can all have large similarities. Require relative semantic evidence.
        anchor_supports = (
            anchor_prob_all.argmax(dim=1).eq(c)
            & (anchor_prob_all[:, c] >= args.anchor_min_confidence)
        )
        prototype_supports = (
            proto_pred_all.eq(c) & (proto_margin_all >= args.prototype_min_margin)
        )
        semantic_support = anchor_supports | prototype_supports
        # Overriding teacher top-1 needs BOTH source classifier/prototype votes.
        semantic_support &= top1_label.eq(c) | (anchor_supports & prototype_supports)
        valid = (
            (~trusted)
            & semantic_support
            & (stable_prob_all[:, c] >= args.recovery_teacher_min_prob)
            & (proto_sim_all[:, c] >= args.recovery_proto_min_similarity)
            & (teacher_rank_all[:, c] <= args.recovery_teacher_topk)
            & (recovery_score[:, c] >= args.recovery_min_score)
        )
        candidate_idx = valid.nonzero(as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            continue
        k = min(args.recovery_topk_per_class, int(candidate_idx.numel()))
        keep = torch.topk(recovery_score[candidate_idx, c], k=k, largest=True).indices
        for sample in candidate_idx[keep].tolist():
            proposals.append((float(recovery_score[sample, c]), sample, c))

    proposals.sort(key=lambda item: item[0], reverse=True)
    recovery_quota = [0] * 7
    for score, sample, c in proposals:
        if proposed_label[sample] >= 0:
            continue
        if recovery_quota[c] >= args.recovery_topk_per_class:
            continue
        proposed_label[sample] = c
        proposed_score[sample] = score * args.recovery_weight_scale
        proposed_recovery[sample] = True
        recovery_quota[c] += 1

    recovery_counts = torch.zeros(7, dtype=torch.long)
    for c in range(7):
        recovery_counts[c] = int((proposed_recovery & proposed_label.eq(c)).sum().item())

    has_proposal = proposed_label.ge(0)
    same = temporal_label.eq(proposed_label)
    new_streak = torch.where(
        has_proposal,
        torch.where(same, temporal_streak + 1, torch.ones_like(temporal_streak)),
        torch.zeros_like(temporal_streak),
    )
    temporal_label.copy_(torch.where(
        has_proposal, proposed_label, torch.full_like(proposed_label, -1)
    ))
    temporal_streak.copy_(new_streak)

    temporal_factor = (
        new_streak.float() / float(max(args.temporal_full_streak, 1))
    ).clamp(0.0, 1.0)
    proposed_score = proposed_score * temporal_factor
    eligible = has_proposal & (new_streak >= args.temporal_min_streak)

    eligible_counts = torch.zeros(7, dtype=torch.long)
    for c in range(7):
        eligible_counts[c] = int((eligible & proposed_label.eq(c)).sum().item())

    # Source-feature geometric votes estimate relative class coverage without
    # reading target labels. Do not force 1,200 samples into every class.
    prototype_counts = torch.bincount(proto_pred_all, minlength=7)
    class_caps = (prototype_counts.float() * args.pseudo_keep_fraction).long()
    class_caps.clamp_(args.class_balance_floor, args.class_balance_max)
    if previous_counts is not None and int(previous_counts.sum()) > 0:
        growth_caps = (previous_counts.float() * args.class_growth_factor).long() + 32
        class_caps = torch.minimum(class_caps, growth_caps)
    class_cap = int(class_caps.max())

    selected = torch.zeros(n_samples, dtype=torch.bool)
    for c in range(7):
        class_idx = (eligible & proposed_label.eq(c)).nonzero(as_tuple=False).flatten()
        if class_idx.numel() == 0:
            continue
        if class_idx.numel() > int(class_caps[c]):
            keep = torch.topk(proposed_score[class_idx], k=int(class_caps[c]), largest=True).indices
            class_idx = class_idx[keep]
        selected[class_idx] = True

    selected_counts = torch.zeros(7, dtype=torch.long)
    for c in range(7):
        selected_counts[c] = int((selected & proposed_label.eq(c)).sum().item())

    class_weight = torch.ones(7, dtype=torch.float32)
    valid_counts = selected_counts > 0
    if valid_counts.any():
        mean_count = selected_counts[valid_counts].float().mean()
        class_weight[valid_counts] = (
            mean_count / selected_counts[valid_counts].float().clamp_min(1.0)
        ).pow(args.class_reweight_power).clamp(args.min_class_weight, args.max_class_weight)

    epoch_label = torch.full((n_samples,), -1, dtype=torch.long)
    epoch_label[selected] = proposed_label[selected]
    epoch_weight = torch.zeros(n_samples, dtype=torch.float32)
    epoch_weight[selected] = (
        proposed_score[selected] * class_weight[proposed_label[selected]]
    )
    epoch_weight.clamp_(0.0, 1.5)
    # Recovery labels may train the classifier cautiously, but must not pull
    # the source/target feature distributions together until independently trusted.
    epoch_align = selected & trusted & (epoch_weight >= args.align_min_weight)

    align_counts = torch.zeros(7, dtype=torch.long)
    for c in range(7):
        align_counts[c] = int((epoch_align & epoch_label.eq(c)).sum().item())
    ddrl_classes = int((align_counts >= args.ddrl_min_class_samples).sum().item())
    ddrl_ready = ddrl_classes >= args.ddrl_min_classes

    return {
        "label": epoch_label,
        "soft_label": torch.where(proposed_recovery.unsqueeze(1),
                                  F.one_hot(proposed_label.clamp_min(0), 7).float(), avg_prob_all),
        "weight": epoch_weight,
        "selected": selected,
        "align": epoch_align,
        "trusted_counts": trusted_counts,
        "recovery_counts": recovery_counts,
        "recovery_needed": recovery_needed,
        "eligible_counts": eligible_counts,
        "selected_counts": selected_counts,
        "align_counts": align_counts,
        "class_cap": class_cap,
        "class_caps": class_caps,
        "prototype_counts": prototype_counts,
        "ddrl_ready": ddrl_ready,
        "ddrl_classes": ddrl_classes,
    }


def scheduled_affinity_weight(epoch, args):
    if args.target_w2 <= 0 or epoch < args.affinity_warmup:
        return 0.0
    progress = float(epoch - args.affinity_warmup + 1) / float(max(args.affinity_ramp, 1))
    return args.target_w2 * min(1.0, progress)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CAST RAF-DB -> FER2013, ResNet50 accuracy-focused training"
    )
    parser.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50", "mobilenet_v2"])
    parser.add_argument("-c", "--checkpoint", default=None)
    parser.add_argument("--source_root", default="/workspace/ttt/code/test-upload-clean/datesets/raf-basic")
    parser.add_argument("--target_root", default="/workspace/ttt/code/data/fer2013")
    parser.add_argument("--model_dir", default="./models/cast_resnet50_v5")
    parser.add_argument("--train_source", action="store_true", help="explicitly train a new source model")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--augmentation", choices=["face", "legacy"], default="face")

    parser.add_argument("--source_epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--source_backbone_lr_mult", type=float, default=0.1)
    parser.add_argument("--source_label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--source_lr_gamma", type=float, default=0.95)
    parser.add_argument("--w1", type=float, default=4.0)
    parser.add_argument("--w2", type=float, default=0.3)
    parser.add_argument("--w3", type=float, default=0.1)

    parser.add_argument("--target_lr", type=float, default=5e-5)
    parser.add_argument("--target_lr_gamma", type=float, default=0.97)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--target_lambda", type=float, default=0.35)
    parser.add_argument("--pseudo_ramp", type=int, default=8)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)

    parser.add_argument("--bn_adapt_batches", type=int, default=64)
    parser.add_argument("--bn_adapt_momentum", type=float, default=0.03)
    parser.add_argument("--bn_mode", choices=["domain", "frozen"], default="domain")

    parser.add_argument("--phi", type=float, default=1.4)
    parser.add_argument("--teacher_temperature", type=float, default=0.0,
                        help="0: calibrate on source validation; positive: fixed temperature")
    parser.add_argument("--threshold_mode", choices=["quantile", "cast"], default="quantile")
    parser.add_argument("--threshold_momentum", type=float, default=0.8)
    parser.add_argument("--pseudo_min_threshold", type=float, default=0.52)
    parser.add_argument("--pseudo_max_threshold", type=float, default=0.995)
    parser.add_argument("--threshold_quantile", type=float, default=0.60)
    parser.add_argument("--min_margin", type=float, default=0.08)
    parser.add_argument("--max_entropy", type=float, default=0.78)
    parser.add_argument("--anchor_min_confidence", type=float, default=0.38)
    parser.add_argument("--prototype_min_margin", type=float, default=0.01)
    parser.add_argument("--prototype_centers", type=int, default=3)
    parser.add_argument("--anchor_guard_epochs", type=int, default=3)
    parser.add_argument("--temporal_min_streak", type=int, default=2)
    parser.add_argument("--temporal_full_streak", type=int, default=3)

    parser.add_argument("--starvation_support", type=int, default=64)
    parser.add_argument("--recovery_trigger_count", type=int, default=96)
    parser.add_argument("--recovery_topk_per_class", type=int, default=0)
    parser.add_argument("--recovery_teacher_topk", type=int, default=2)
    parser.add_argument("--recovery_teacher_min_prob", type=float, default=0.12)
    parser.add_argument("--recovery_proto_min_similarity", type=float, default=0.45)
    parser.add_argument("--recovery_min_score", type=float, default=0.40)
    parser.add_argument("--recovery_weight_scale", type=float, default=0.70)
    parser.add_argument("--min_pseudo_batch", type=int, default=8)

    parser.add_argument("--class_balance_floor", type=int, default=32)
    parser.add_argument("--class_balance_max", type=int, default=4000)
    parser.add_argument("--class_balance_factor", type=float, default=None,
                        help="deprecated; use pseudo_keep_fraction and class_growth_factor")
    parser.add_argument("--pseudo_keep_fraction", type=float, default=0.4)
    parser.add_argument("--class_growth_factor", type=float, default=1.3)
    parser.add_argument("--class_reweight_power", type=float, default=0.0)
    parser.add_argument("--min_class_weight", type=float, default=0.65)
    parser.add_argument("--max_class_weight", type=float, default=1.50)

    parser.add_argument("--target_w2", type=float, default=0.0)
    parser.add_argument("--affinity_warmup", type=int, default=6)
    parser.add_argument("--affinity_ramp", type=int, default=6)
    parser.add_argument("--align_min_weight", type=float, default=0.30)
    parser.add_argument("--ddrl_min_classes", type=int, default=3)
    parser.add_argument("--ddrl_min_class_samples", type=int, default=12)

    args = parser.parse_args()
    if args.class_balance_factor is not None:
        print("class_balance_factor is deprecated and ignored; using geometric class quotas")
    if not 0.0 <= args.pseudo_min_threshold <= args.pseudo_max_threshold < 1.0:
        parser.error("invalid pseudo-label threshold range")
    if args.temporal_min_streak < 1 or args.temporal_full_streak < args.temporal_min_streak:
        parser.error("invalid temporal streak configuration")
    if not 1 <= args.recovery_teacher_topk <= 7:
        parser.error("recovery_teacher_topk must be in [1, 7]")
    if args.class_balance_floor < 1 or args.class_balance_max < args.class_balance_floor:
        parser.error("invalid class balance cap")
    if not 0 <= args.threshold_momentum < 1 or not 0 <= args.threshold_quantile <= 1:
        parser.error("invalid threshold quantile/momentum")
    if args.teacher_temperature < 0 or args.min_pseudo_batch < 1:
        parser.error("temperature must be nonnegative; min_pseudo_batch must be positive")
    if args.batch_size < 2 or args.epochs < 1:
        parser.error("batch_size must be >= 2 and epochs must be positive")
    if not 0 < args.bn_adapt_momentum <= 1 or args.grad_clip <= 0:
        parser.error("invalid BN momentum or gradient clip")
    if args.source_backbone_lr_mult <= 0:
        parser.error("source_backbone_lr_mult must be positive")
    if not args.checkpoint and not args.train_source:
        parser.error("provide --checkpoint SOURCE.pth, or explicitly pass --train_source")
    if args.checkpoint and args.train_source:
        parser.error("choose checkpoint initialization OR training a new source model")
    if args.train_source and args.source_epochs < 1:
        parser.error("source_epochs must be positive")
    if args.checkpoint and not os.path.isfile(args.checkpoint):
        parser.error("source checkpoint not found: %s" % args.checkpoint)
    if not 0 <= args.source_label_smoothing < 1 or args.prototype_centers < 1:
        parser.error("invalid label smoothing or prototype_centers")
    if not 0 < args.pseudo_keep_fraction <= 1 or args.class_growth_factor < 1:
        parser.error("invalid pseudo-label keep fraction or class growth factor")
    if args.recovery_topk_per_class < 0 or args.class_reweight_power < 0:
        parser.error("recovery count and class_reweight_power must be nonnegative")
    return args


def train_source(model, loader, val_loader, args, source_path):
    optimizer = torch.optim.Adam(
        optimizer_groups(model, args.lr, args.source_backbone_lr_mult), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.source_lr_gamma)
    source_best_path = source_path.replace("_source_final.pth", "_source_best.pth")
    best_source = -1.0

    for epoch in range(args.source_epochs):
        model.train()
        correct = 0
        seen = 0
        cls_sum = 0.0
        aff_sum = 0.0
        steps = 0

        for imgs, targets in loader:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.long().cuda(non_blocking=True)
            output = model(
                imgs, targets, None, "train", "source", compute_affinity=args.w2 > 0
            )
            cls_loss = smoothed_cross_entropy(output[0], targets, args.source_label_smoothing)
            aff_loss = output[1]
            loss = (
                args.w1 * cls_loss
                + args.w2 * aff_loss
                + args.w3 * classifier_weight_loss(model)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite source loss")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            correct += output[0].argmax(dim=1).eq(targets).sum().item()
            seen += targets.numel()
            cls_sum += float(cls_loss.detach())
            aff_sum += float(aff_loss.detach())
            steps += 1

        scheduler.step()
        print(
            "[Source %d] acc %.4f cls %.4f aff %.4f lr %.6f"
            % (
                epoch,
                correct / float(max(seen, 1)),
                cls_sum / max(steps, 1),
                aff_sum / max(steps, 1),
                optimizer.param_groups[0]["lr"],
            )
        )
        val_acc = evaluate(model, val_loader, len(val_loader.dataset), task="source", verbose=False)
        print("[Source %d] source_validation_accuracy %.4f" % (epoch, val_acc))
        if val_acc > best_source:
            best_source = val_acc
            save_checkpoint(model, optimizer, source_best_path, epoch, val_acc, "source_best")

    save_checkpoint(model, optimizer, source_path, args.source_epochs - 1, val_acc, "source")
    print("Source checkpoint saved:", source_path)
    model.load_state_dict(torch.load(source_best_path, map_location="cuda")["model"], strict=True)
    print("Use source-validation checkpoint:", source_best_path)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    print("CAST v5 configuration:", json.dumps(vars(args), sort_keys=True))
    tx = make_transforms(args.augmentation)

    source_train = RafDataSet(
        args.source_root, "train", transform=tx["source"], strong_transform=None, basic_aug=False
    )
    source_proto = RafDataSet(
        args.source_root, "train", transform=tx["test"], strong_transform=None, basic_aug=False
    )
    source_val = RafDataSet(args.source_root, "test", transform=tx["test"], strong_transform=None)
    target_base = FER(
        args.target_root, "train", transform=tx["weak"], strong_transform=tx["strong"], basic_aug=False
    )
    target_train = IndexedTargetDataset(target_base)
    target_scan_base = FER(
        args.target_root, "train", transform=tx["test"], strong_transform=None, basic_aug=False
    )
    if target_scan_base.file_paths != target_base.file_paths:
        raise RuntimeError("Target scan and training sample indices differ")
    target_scan = TargetScanDataset(target_scan_base)
    target_test = FER(
        args.target_root, "test", transform=tx["test"], strong_transform=None
    )

    source_loader = make_loader(source_train, args.batch_size, args.workers, True, True, 1)
    proto_loader = make_loader(source_proto, args.batch_size, args.workers, False, False, 2)
    target_loader = make_loader(target_train, args.batch_size, args.workers, True, True, 3)
    target_scan_loader = make_loader(target_scan, args.batch_size, args.workers, False, False, 4)
    test_loader = make_loader(target_test, args.batch_size, args.workers, False, False, 5)
    source_val_loader = make_loader(source_val, args.batch_size, args.workers, False, False, 6)
    if len(source_loader) == 0 or len(target_loader) == 0 or len(target_test) == 0 or len(source_val) == 0:
        raise RuntimeError("Empty loader: check dataset paths, split and batch_size")

    model = Networks.Model(backbone=args.backbone, num_classes=7,
                           pretrained=args.checkpoint is None).cuda()
    prefix = "%s_rafdb_fer" % args.backbone
    source_path = os.path.join(args.model_dir, prefix + "_source_final.pth")
    student_best_path = os.path.join(args.model_dir, prefix + "_student_best.pth")
    ema_best_path = os.path.join(args.model_dir, prefix + "_ema_best.pth")
    best_path = os.path.join(args.model_dir, prefix + "_best.pth")
    student_final_path = os.path.join(args.model_dir, prefix + "_student_final.pth")
    ema_final_path = os.path.join(args.model_dir, prefix + "_ema_final.pth")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        model.load_state_dict(checkpoint["model"], strict=True)
        print("Loaded source checkpoint:", args.checkpoint)
    else:
        train_source(model, source_loader, source_val_loader, args, source_path)

    if args.teacher_temperature == 0.0:
        args.teacher_temperature = calibrate_temperature(model, source_val_loader)
    with open(os.path.join(args.model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    source_target_acc = evaluate(model, test_loader, len(target_test))
    print("[Source -> Target] accuracy %.4f" % source_target_acc)

    anchor = copy.deepcopy(model).cuda().eval()
    for parameter in anchor.parameters():
        parameter.requires_grad = False
    prototypes, prototype_counts = build_source_prototypes(
        anchor, proto_loader, centers_per_class=args.prototype_centers
    )
    print("Source prototype counts:", prototype_counts.tolist())

    if args.bn_mode == "domain":
        enable_domain_bn(model, momentum=args.bn_adapt_momentum)

    adapt_backbone_bn(
        model,
        target_scan_loader,
        args.bn_adapt_batches,
        args.bn_adapt_momentum,
    )
    teacher = copy.deepcopy(model).cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    target_start_acc = evaluate(model, test_loader, len(target_test))
    print("[Target start after BN] accuracy %.4f" % target_start_acc)

    optimizer = torch.optim.Adam(
        optimizer_groups(model, args.target_lr, 0.20 if args.backbone == "resnet50" else 0.35),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.target_lr_gamma)

    temporal_label = torch.full((len(target_train),), -1, dtype=torch.long)
    temporal_streak = torch.zeros(len(target_train), dtype=torch.long)
    previous_thresholds = None
    previous_selected_counts = None

    best_student = target_start_acc
    best_ema = target_start_acc
    best_overall = target_start_acc
    save_checkpoint(model, optimizer, student_best_path, -1, target_start_acc, "student")
    save_checkpoint(teacher, None, ema_best_path, -1, target_start_acc, "ema")
    save_checkpoint(model, optimizer, best_path, -1, target_start_acc, "target_bn")

    source_iter = iter(source_loader)

    for epoch in range(args.epochs):
        freeze_backbone = epoch < args.freeze_backbone_epochs
        for parameter in model.feature.parameters():
            parameter.requires_grad = not freeze_backbone

        (
            thresholds,
            threshold_support,
            predicted_count,
            weak_agreement,
            anchor_agreement,
        ) = estimate_target_thresholds(
            teacher, anchor, target_scan_loader, args, epoch, previous_thresholds
        )
        previous_thresholds = thresholds.clone()

        pseudo_bank = build_epoch_pseudo_bank(
            teacher,
            anchor,
            target_scan_loader,
            prototypes,
            thresholds,
            threshold_support,
            temporal_label,
            temporal_streak,
            args,
            epoch,
            previous_selected_counts,
        )
        previous_selected_counts = pseudo_bank["selected_counts"].clone()
        print("[Epoch %d] prototype_votes %s class_caps %s" % (
            epoch, pseudo_bank["prototype_counts"].tolist(), pseudo_bank["class_caps"].tolist()))

        pseudo_scale = min(1.0, float(epoch + 1) / float(max(args.pseudo_ramp, 1)))
        scheduled_w2 = scheduled_affinity_weight(epoch, args)
        active_w2 = scheduled_w2 if pseudo_bank["ddrl_ready"] else 0.0

        print(
            "[Epoch %d] CATM %s support %s predicted %s weak_agree %.4f anchor_agree %.4f "
            "trusted %s recovery %s need_recovery %s cap %d eligible %s selected %s align %s "
            "ddrl_classes %d w2 %.4f pseudo_scale %.3f"
            % (
                epoch,
                [round(float(x), 4) for x in thresholds.tolist()],
                threshold_support.tolist(),
                predicted_count.tolist(),
                weak_agreement,
                anchor_agreement,
                pseudo_bank["trusted_counts"].tolist(),
                pseudo_bank["recovery_counts"].tolist(),
                pseudo_bank["recovery_needed"].int().tolist(),
                pseudo_bank["class_cap"],
                pseudo_bank["eligible_counts"].tolist(),
                pseudo_bank["selected_counts"].tolist(),
                pseudo_bank["align_counts"].tolist(),
                pseudo_bank["ddrl_classes"],
                active_w2,
                pseudo_scale,
            )
        )

        selected_total = 0
        pseudo_correct = 0
        class_total = np.zeros(7, dtype=np.int64)
        class_correct = np.zeros(7, dtype=np.int64)
        source_ce_sum = 0.0
        target_ce_sum = 0.0
        applied_w2_sum = 0.0
        steps = 0

        for weak1, _, strong, gt_target, sample_idx in target_loader:
            try:
                source_imgs, source_targets = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_imgs, source_targets = next(source_iter)

            idx = sample_idx.long().cpu()
            pseudo_targets = pseudo_bank["label"][idx]
            selected = pseudo_targets.ge(0)
            pseudo_weight = pseudo_bank["weight"][idx]
            align_mask = pseudo_bank["align"][idx]
            selected_n = int(selected.sum().item())
            selected_total += selected_n

            gt_cpu = gt_target.long().cpu()
            if selected.any():
                pseudo_correct += int(pseudo_targets[selected].eq(gt_cpu[selected]).sum().item())
                for c in range(7):
                    cm = selected & pseudo_targets.eq(c)
                    n = int(cm.sum().item())
                    if n:
                        class_total[c] += n
                        class_correct[c] += int(pseudo_targets[cm].eq(gt_cpu[cm]).sum().item())

            if args.bn_mode == "domain":
                refresh_target_bn(model, weak1.cuda(non_blocking=True))
            model.train()
            if freeze_backbone:
                model.feature.eval()
            # Domain mode refreshes target moments on weak images each step;
            # frozen mode retains the v4 fixed-statistics behavior for ablation.
            if args.bn_mode == "frozen":
                freeze_backbone_bn_stats(model)

            source_imgs = source_imgs.cuda(non_blocking=True)
            source_targets = source_targets.long().cuda(non_blocking=True)
            strong = strong.cuda(non_blocking=True)
            selected_cuda = selected.cuda(non_blocking=True)
            weight_cuda = pseudo_weight.cuda(non_blocking=True)
            align_cuda = align_mask.cuda(non_blocking=True)
            safe_target = pseudo_targets.clamp_min(0).long().cuda(non_blocking=True)

            n_source = source_imgs.shape[0]
            train_imgs = torch.cat((source_imgs, strong), dim=0)
            train_targets = torch.cat((source_targets, safe_target), dim=0)
            affinity_mask = torch.cat((
                torch.ones(n_source, dtype=torch.bool, device="cuda"),
                align_cuda,
            ), dim=0)

            source_classes = torch.bincount(source_targets, minlength=7) > 0
            target_counts = torch.bincount(safe_target[align_cuda], minlength=7)
            supported_classes = source_classes & (target_counts >= 2)
            batch_w2 = active_w2 if int(supported_classes.sum().item()) >= 2 else 0.0
            output = model(
                train_imgs,
                train_targets,
                affinity_mask,
                "train",
                "target",
                source_count=n_source,
                compute_affinity=batch_w2 > 0,
            )

            source_loss = smoothed_cross_entropy(
                output[0][:n_source], source_targets, args.source_label_smoothing
            )
            if selected_n > 0:
                target_logits = output[0][n_source:][selected_cuda]
                soft_labels = pseudo_bank["soft_label"][idx][selected].cuda(non_blocking=True)
                temperature = args.teacher_temperature
                per_target = F.kl_div(
                    F.log_softmax(target_logits / temperature, dim=1),
                    soft_labels, reduction="none"
                ).sum(dim=1) * temperature ** 2
                selected_weight = weight_cuda[selected_cuda]
                # Keep absolute reliability: dividing by sum(weights) cancels
                # the recovery discount when a batch contains only recovery
                # samples. A small support floor also limits sparse-batch noise.
                target_loss = (
                    per_target * selected_weight
                ).sum() / float(max(selected_n, args.min_pseudo_batch))
            else:
                target_loss = source_loss.new_zeros(())

            classification_loss = source_loss + args.target_lambda * pseudo_scale * target_loss
            loss = (
                args.w1 * classification_loss
                + batch_w2 * output[1]
                + args.w3 * classifier_weight_loss(model)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite target loss")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            update_ema(model, teacher, args.ema_decay)

            source_ce_sum += float(source_loss.detach())
            target_ce_sum += float(target_loss.detach())
            applied_w2_sum += batch_w2
            steps += 1

        scheduler.step()
        pseudo_acc = pseudo_correct / float(max(selected_total, 1))
        class_acc = [
            round(class_correct[c] / float(class_total[c]), 4) if class_total[c] else 0.0
            for c in range(7)
        ]
        print(
            "[Epoch %d] selected %d pseudo_acc %.4f class_acc %s class_num %s "
            "source_ce %.4f target_kl %.4f applied_w2 %.4f"
            % (
                epoch,
                selected_total,
                pseudo_acc,
                class_acc,
                class_total.tolist(),
                source_ce_sum / max(steps, 1),
                target_ce_sum / max(steps, 1),
                applied_w2_sum / max(steps, 1),
            )
        )

        student_acc = evaluate(model, test_loader, len(target_test))
        ema_acc = evaluate(teacher, test_loader, len(target_test))
        print("[Epoch %d] Student accuracy: %.4f | EMA accuracy: %.4f" % (epoch, student_acc, ema_acc))

        if student_acc > best_student:
            best_student = student_acc
            save_checkpoint(model, optimizer, student_best_path, epoch, student_acc, "student")
        if ema_acc > best_ema:
            best_ema = ema_acc
            save_checkpoint(teacher, None, ema_best_path, epoch, ema_acc, "ema")
        if max(student_acc, ema_acc) > best_overall:
            if ema_acc >= student_acc:
                best_overall = ema_acc
                save_checkpoint(teacher, None, best_path, epoch, ema_acc, "ema")
            else:
                best_overall = student_acc
                save_checkpoint(model, optimizer, best_path, epoch, student_acc, "student")
            print("Best checkpoint:", best_path, "acc", best_overall)

    final_student = evaluate(model, test_loader, len(target_test))
    final_ema = evaluate(teacher, test_loader, len(target_test))
    save_checkpoint(model, optimizer, student_final_path, args.epochs - 1, final_student, "student_final")
    save_checkpoint(teacher, None, ema_final_path, args.epochs - 1, final_ema, "ema_final")
    print(
        "best_student %.4f best_ema %.4f best_overall %.4f final_student %.4f final_ema %.4f"
        % (best_student, best_ema, best_overall, final_student, final_ema)
    )


if __name__ == "__main__":
    main()
