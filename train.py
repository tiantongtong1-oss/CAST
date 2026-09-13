import argparse
import copy
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


def make_transforms():
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
    return {"source": source, "weak": weak, "strong": strong, "test": test}


def classifier_weight_loss(model):
    weight = F.normalize(model.fc.weight, dim=1)
    identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
    return ((weight.mm(weight.t()) - identity + 1.0) / 2.0).mean()


def freeze_backbone_bn_stats(model):
    """Keep the target-recalibrated ResNet BN running statistics fixed."""
    for module in model.feature.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


@torch.no_grad()
def update_ema(student, teacher, decay):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if teacher_buffer.dtype.is_floating_point:
            teacher_buffer.data.mul_(decay).add_(student_buffer.data, alpha=1.0 - decay)
        else:
            teacher_buffer.data.copy_(student_buffer.data)


@torch.no_grad()
def evaluate(model, loader, num):
    model.eval()
    correct = 0
    preds, labels = [], []
    for imgs, targets in loader:
        imgs = imgs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        logits, _ = model(imgs, targets, None, mode="test")
        pred = logits.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        preds.append(pred.cpu())
        labels.append(targets.cpu())
    util.make_confucion_matrix(preds, labels)
    return float(np.around(correct / float(max(num, 1)), 4))


def save_checkpoint(model, optimizer, path, epoch, accuracy, kind):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "accuracy": accuracy,
            "model_kind": kind,
        },
        path,
    )


@torch.no_grad()
def adapt_backbone_bn(model, target_loader, max_batches, momentum):
    """Unsupervised target-domain BN recalibration for the pretrained backbone."""
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
    print("Target BN recalibration batches:", used)


@torch.no_grad()
def build_source_prototypes(anchor, loader, class_num=7):
    anchor.eval()
    sums = None
    counts = torch.zeros(class_num, dtype=torch.long)

    for imgs, targets in loader:
        imgs = imgs.cuda(non_blocking=True)
        _, features = anchor(imgs, None, None, mode="test", task="source")
        features = F.normalize(features.float(), dim=1)
        targets = targets.long().cpu()

        if sums is None:
            sums = torch.zeros(class_num, features.shape[1])
        for c in range(class_num):
            mask = targets.eq(c)
            if mask.any():
                sums[c] += features[mask].sum(dim=0)
                counts[c] += int(mask.sum().item())

    if sums is None or (counts == 0).any():
        raise RuntimeError("Incomplete source prototypes: %s" % counts.tolist())

    prototypes = F.normalize(
        sums / counts.float().unsqueeze(1).clamp_min(1.0), dim=1
    )
    return prototypes, counts


def tempered_probability(logits, temperature):
    return F.softmax(logits / float(max(temperature, 1e-6)), dim=1)


@torch.no_grad()
def estimate_target_thresholds(teacher, anchor, loader, args, epoch):
    teacher.eval()
    anchor.eval()
    class_all = [[] for _ in range(7)]
    class_anchor = [[] for _ in range(7)]
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

        features = F.normalize(anchor_features.float(), dim=1)
        similarity = features.mm(prototypes.t())
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
            trusted_batch = base & (anchor_cpu | proto_match)

        proto_quality = ((proto_margin - args.prototype_min_margin) / 0.35).clamp(0.0, 1.0)
        quality = (
            0.45 * stable_top1.cpu()
            + 0.20 * margin.cpu()
            + 0.15 * (1.0 - entropy.cpu())
            + 0.10 * anchor_conf.cpu()
            + 0.10 * proto_quality
        )

        avg_cpu = avg.cpu()
        rank = (avg_cpu.unsqueeze(1) < avg_cpu.unsqueeze(2)).sum(dim=2) + 1

        top1_label[idx] = target.cpu()
        trusted[idx] = trusted_batch
        trusted_score[idx] = quality
        avg_prob_all[idx] = avg_cpu
        stable_prob_all[idx] = stable_all.cpu()
        anchor_prob_all[idx] = anchor_prob.cpu()
        proto_sim_all[idx] = ((similarity.cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
        teacher_rank_all[idx] = rank.long()

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
        valid = (
            (~trusted)
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

    nonzero = eligible_counts[eligible_counts > 0].float()
    if nonzero.numel():
        median_count = float(nonzero.median().item())
        class_cap = int(max(
            args.class_balance_floor,
            min(args.class_balance_max, median_count * args.class_balance_factor),
        ))
    else:
        class_cap = args.class_balance_floor

    selected = torch.zeros(n_samples, dtype=torch.bool)
    for c in range(7):
        class_idx = (eligible & proposed_label.eq(c)).nonzero(as_tuple=False).flatten()
        if class_idx.numel() == 0:
            continue
        if class_idx.numel() > class_cap:
            keep = torch.topk(proposed_score[class_idx], k=class_cap, largest=True).indices
            class_idx = class_idx[keep]
        selected[class_idx] = True

    selected_counts = torch.zeros(7, dtype=torch.long)
    for c in range(7):
        selected_counts[c] = int((selected & proposed_label.eq(c)).sum().item())

    class_weight = torch.ones(7, dtype=torch.float32)
    valid_counts = selected_counts > 0
    if valid_counts.any():
        mean_count = selected_counts[valid_counts].float().mean()
        class_weight[valid_counts] = torch.sqrt(
            mean_count / selected_counts[valid_counts].float().clamp_min(1.0)
        ).clamp(args.min_class_weight, args.max_class_weight)

    epoch_label = torch.full((n_samples,), -1, dtype=torch.long)
    epoch_label[selected] = proposed_label[selected]
    epoch_weight = torch.zeros(n_samples, dtype=torch.float32)
    epoch_weight[selected] = (
        proposed_score[selected] * class_weight[proposed_label[selected]]
    )
    epoch_weight.clamp_(0.0, 1.5)
    epoch_align = selected & (epoch_weight >= args.align_min_weight)

    align_counts = torch.zeros(7, dtype=torch.long)
    for c in range(7):
        align_counts[c] = int((epoch_align & epoch_label.eq(c)).sum().item())
    ddrl_classes = int((align_counts >= args.ddrl_min_class_samples).sum().item())
    ddrl_ready = ddrl_classes >= args.ddrl_min_classes

    return {
        "label": epoch_label,
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
    parser.add_argument("--model_dir", default="./models/cast_resnet50_v3")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--source_epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    parser.add_argument("--phi", type=float, default=1.4)
    parser.add_argument("--teacher_temperature", type=float, default=1.4)
    parser.add_argument("--pseudo_min_threshold", type=float, default=0.52)
    parser.add_argument("--pseudo_max_threshold", type=float, default=0.90)
    parser.add_argument("--threshold_quantile", type=float, default=0.60)
    parser.add_argument("--min_margin", type=float, default=0.08)
    parser.add_argument("--max_entropy", type=float, default=0.78)
    parser.add_argument("--anchor_min_confidence", type=float, default=0.38)
    parser.add_argument("--prototype_min_margin", type=float, default=0.01)
    parser.add_argument("--anchor_guard_epochs", type=int, default=3)
    parser.add_argument("--temporal_min_streak", type=int, default=2)
    parser.add_argument("--temporal_full_streak", type=int, default=3)

    parser.add_argument("--starvation_support", type=int, default=64)
    parser.add_argument("--recovery_trigger_count", type=int, default=96)
    parser.add_argument("--recovery_topk_per_class", type=int, default=128)
    parser.add_argument("--recovery_teacher_topk", type=int, default=4)
    parser.add_argument("--recovery_teacher_min_prob", type=float, default=0.12)
    parser.add_argument("--recovery_proto_min_similarity", type=float, default=0.45)
    parser.add_argument("--recovery_min_score", type=float, default=0.40)
    parser.add_argument("--recovery_weight_scale", type=float, default=0.70)

    parser.add_argument("--class_balance_floor", type=int, default=256)
    parser.add_argument("--class_balance_max", type=int, default=1200)
    parser.add_argument("--class_balance_factor", type=float, default=3.0)
    parser.add_argument("--min_class_weight", type=float, default=0.65)
    parser.add_argument("--max_class_weight", type=float, default=1.50)

    parser.add_argument("--target_w2", type=float, default=0.05)
    parser.add_argument("--affinity_warmup", type=int, default=6)
    parser.add_argument("--affinity_ramp", type=int, default=6)
    parser.add_argument("--align_min_weight", type=float, default=0.30)
    parser.add_argument("--ddrl_min_classes", type=int, default=3)
    parser.add_argument("--ddrl_min_class_samples", type=int, default=12)

    args = parser.parse_args()
    if not 0.0 <= args.pseudo_min_threshold <= args.pseudo_max_threshold <= 0.99:
        parser.error("invalid pseudo-label threshold range")
    if args.temporal_min_streak < 1 or args.temporal_full_streak < args.temporal_min_streak:
        parser.error("invalid temporal streak configuration")
    if not 1 <= args.recovery_teacher_topk <= 7:
        parser.error("recovery_teacher_topk must be in [1, 7]")
    if args.class_balance_floor < 1 or args.class_balance_max < args.class_balance_floor:
        parser.error("invalid class balance cap")
    return args


def train_source(model, loader, args, source_path):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.source_lr_gamma)

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
            cls_loss = criterion(output[0], targets).mean()
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

    save_checkpoint(model, optimizer, source_path, args.source_epochs - 1, 0.0, "source")
    print("Source checkpoint saved:", source_path)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    tx = make_transforms()

    source_train = RafDataSet(
        args.source_root, "train", transform=tx["source"], strong_transform=None, basic_aug=False
    )
    source_proto = RafDataSet(
        args.source_root, "train", transform=tx["test"], strong_transform=None, basic_aug=False
    )
    target_base = FER(
        args.target_root, "train", transform=tx["weak"], strong_transform=tx["strong"], basic_aug=False
    )
    target_train = IndexedTargetDataset(target_base)
    target_test = FER(
        args.target_root, "test", transform=tx["test"], strong_transform=None
    )

    source_loader = make_loader(source_train, args.batch_size, args.workers, True, True, 1)
    proto_loader = make_loader(source_proto, args.batch_size, args.workers, False, False, 2)
    target_loader = make_loader(target_train, args.batch_size, args.workers, True, True, 3)
    target_scan_loader = make_loader(target_train, args.batch_size, args.workers, False, False, 4)
    test_loader = make_loader(target_test, args.batch_size, args.workers, False, False, 5)

    model = Networks.Model(backbone=args.backbone, num_classes=7).cuda()
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
        train_source(model, source_loader, args, source_path)

    source_target_acc = evaluate(model, test_loader, len(target_test))
    print("[Source -> Target] accuracy %.4f" % source_target_acc)

    anchor = copy.deepcopy(model).cuda().eval()
    for parameter in anchor.parameters():
        parameter.requires_grad = False
    prototypes, prototype_counts = build_source_prototypes(anchor, proto_loader)
    print("Source prototype counts:", prototype_counts.tolist())

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

    feature_lr = args.target_lr * (0.20 if args.backbone == "resnet50" else 0.35)
    optimizer = torch.optim.Adam(
        [
            {"params": model.feature.parameters(), "lr": feature_lr},
            {"params": model.fc.parameters(), "lr": args.target_lr},
            {"params": model.bn.parameters(), "lr": args.target_lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.target_lr_gamma)
    source_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    target_criterion = torch.nn.CrossEntropyLoss(reduction="none")

    temporal_label = torch.full((len(target_train),), -1, dtype=torch.long)
    temporal_streak = torch.zeros(len(target_train), dtype=torch.long)

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
        ) = estimate_target_thresholds(teacher, anchor, target_scan_loader, args, epoch)

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
        )

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

        for _, _, strong, gt_target, sample_idx in target_loader:
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

            model.train()
            # Preserve the target-domain BN calibration while still updating convolutional weights.
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

            batch_w2 = active_w2 if int(align_cuda.sum().item()) > 0 else 0.0
            output = model(
                train_imgs,
                train_targets,
                affinity_mask,
                "train",
                "target",
                source_count=n_source,
                compute_affinity=batch_w2 > 0,
            )

            source_loss = source_criterion(output[0][:n_source], source_targets).mean()
            if selected_n > 0:
                target_logits = output[0][n_source:][selected_cuda]
                target_labels = safe_target[selected_cuda]
                per_target = target_criterion(target_logits, target_labels)
                selected_weight = weight_cuda[selected_cuda]
                target_loss = (
                    per_target * selected_weight
                ).sum() / selected_weight.sum().clamp_min(1.0)
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
            "source_ce %.4f target_ce %.4f applied_w2 %.4f"
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
