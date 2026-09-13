import argparse
import copy
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
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
    """Add a stable sample index for the temporal pseudo-label bank."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if len(item) != 4:
            raise RuntimeError("FER target training dataset must return 4 values")
        return item[0], item[1], item[2], item[3], idx


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
    return {
        "source": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.18)),
        ]),
        "prototype": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]),
        "weak": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        "strong": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.20)),
        ]),
        "test": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]),
    }


def classifier_weight_loss(model):
    normalized = F.normalize(model.fc.weight, dim=1)
    identity = torch.eye(
        normalized.shape[0], device=normalized.device, dtype=normalized.dtype
    )
    return ((normalized.mm(normalized.t()) - identity + 1.0) / 2.0).mean()


def source_class_weights(dataset, device):
    counts = torch.tensor(dataset.label_dis, dtype=torch.float32, device=device)
    mean_count = counts.mean()
    weights = torch.sqrt(mean_count / counts.clamp_min(1.0))
    weights = weights.clamp(0.75, 1.50)
    weights = weights / weights.mean()
    return weights


@torch.no_grad()
def update_ema(student, teacher, decay):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(
            student_param.data, alpha=1.0 - decay
        )
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if teacher_buffer.dtype.is_floating_point:
            teacher_buffer.data.mul_(decay).add_(
                student_buffer.data, alpha=1.0 - decay
            )
        else:
            teacher_buffer.data.copy_(student_buffer.data)


@torch.no_grad()
def evaluate(model, loader, num):
    model.eval()
    correct = 0
    preds = []
    labels = []
    for imgs, targets in loader:
        imgs = imgs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        out, _ = model(imgs, targets, None, mode="test")
        pred = out.argmax(dim=1)
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
                counts[c] += int(mask.sum())

    if sums is None or (counts == 0).any():
        raise RuntimeError(
            "Cannot build complete source prototypes: %s" % counts.tolist()
        )

    prototypes = F.normalize(
        sums / counts.float().unsqueeze(1).clamp_min(1.0), dim=1
    )
    return prototypes, counts


def tempered_probability(logits, temperature):
    return F.softmax(logits / float(max(temperature, 1e-6)), dim=1)


@torch.no_grad()
def estimate_target_thresholds(teacher, anchor, loader, args, epoch):
    """CAST CATM with temperature calibration and agreement-based smoothing."""
    teacher.eval()
    anchor.eval()

    class_all = [[] for _ in range(7)]
    class_stable = [[] for _ in range(7)]
    predicted = torch.zeros(7, dtype=torch.long)
    total = 0
    weak_agree = 0
    anchor_agree = 0

    for imgs_w1, imgs_w2, _, _, _ in loader:
        imgs_w1 = imgs_w1.cuda(non_blocking=True)
        imgs_w2 = imgs_w2.cuda(non_blocking=True)

        out_w1, _ = teacher(imgs_w1, None, None, "test", "target")
        out_w2, _ = teacher(imgs_w2, None, None, "test", "target")
        anchor_out, _ = anchor(imgs_w1, None, None, "test", "target")

        prob_w1 = tempered_probability(out_w1, args.teacher_temperature).cpu()
        prob_w2 = tempered_probability(out_w2, args.teacher_temperature).cpu()
        anchor_prob = tempered_probability(
            anchor_out, args.teacher_temperature
        ).cpu()

        avg_prob = 0.5 * (prob_w1 + prob_w2)
        confidence, targets = avg_prob.max(dim=1)
        pred_w1 = prob_w1.argmax(dim=1)
        pred_w2 = prob_w2.argmax(dim=1)
        agreement = pred_w1.eq(pred_w2)
        anchor_conf, anchor_pred = anchor_prob.max(dim=1)
        anchor_match = anchor_pred.eq(targets) & (
            anchor_conf >= args.anchor_min_confidence
        )

        stable_conf = torch.minimum(
            prob_w1.gather(1, targets.unsqueeze(1)).squeeze(1),
            prob_w2.gather(1, targets.unsqueeze(1)).squeeze(1),
        )

        total += avg_prob.shape[0]
        weak_agree += int(agreement.sum().item())
        anchor_agree += int((agreement & anchor_match).sum().item())
        predicted += torch.bincount(targets, minlength=7)

        for c in range(7):
            values = confidence[targets == c]
            if values.numel() > 0:
                class_all[c].append(values)
            stable_values = stable_conf[
                (targets == c) & agreement & anchor_match
            ]
            if stable_values.numel() > 0:
                class_stable[c].append(stable_values)

    progress = float(epoch) / float(max(args.epochs - 1, 1))
    q = float(np.clip(args.threshold_quantile + 0.05 * progress, 0.0, 0.90))
    stage_factor = float(args.epochs) / float(max(args.epochs - epoch, 1))

    thresholds = torch.full((7,), args.pseudo_max_threshold, dtype=torch.float32)
    support = torch.zeros(7, dtype=torch.long)

    for c in range(7):
        if not class_all[c]:
            continue

        all_values = torch.cat(class_all[c])
        cast_threshold = float(all_values.mean().item()) * args.phi * stage_factor
        cast_threshold = float(
            np.clip(
                cast_threshold,
                args.pseudo_min_threshold,
                args.pseudo_max_threshold,
            )
        )

        threshold = cast_threshold
        if class_stable[c]:
            stable_values = torch.cat(class_stable[c])
            support[c] = stable_values.numel()
            if stable_values.numel() >= 8:
                quantile_threshold = float(torch.quantile(stable_values, q).item())
                threshold = 0.75 * cast_threshold + 0.25 * quantile_threshold
            else:
                threshold = max(cast_threshold, args.pseudo_max_threshold - 0.02)
        else:
            threshold = args.pseudo_max_threshold

        thresholds[c] = float(
            np.clip(
                threshold,
                args.pseudo_min_threshold,
                args.pseudo_max_threshold,
            )
        )

    return (
        thresholds,
        support,
        predicted,
        weak_agree / float(max(total, 1)),
        anchor_agree / float(max(total, 1)),
    )


@torch.no_grad()
def select_pseudo_labels(
    teacher_out_1,
    teacher_out_2,
    anchor_out,
    anchor_features,
    prototypes,
    thresholds,
    sample_idx,
    bank_label,
    bank_streak,
    class_weight,
    args,
    strict_anchor,
):
    prob_1 = tempered_probability(
        teacher_out_1, args.teacher_temperature
    )
    prob_2 = tempered_probability(
        teacher_out_2, args.teacher_temperature
    )
    avg_prob = 0.5 * (prob_1 + prob_2)

    targets = avg_prob.argmax(dim=1)
    agreement = prob_1.argmax(dim=1).eq(prob_2.argmax(dim=1))
    stable_conf = torch.minimum(
        prob_1.gather(1, targets.unsqueeze(1)).squeeze(1),
        prob_2.gather(1, targets.unsqueeze(1)).squeeze(1),
    )
    sample_threshold = thresholds.to(teacher_out_1.device)[targets]

    top2 = torch.topk(avg_prob, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(
        avg_prob.clamp_min(1e-8) * avg_prob.clamp_min(1e-8).log()
    ).sum(dim=1)
    entropy = entropy / np.log(float(avg_prob.shape[1]))

    anchor_prob = tempered_probability(anchor_out, args.teacher_temperature)
    anchor_conf, anchor_pred = anchor_prob.max(dim=1)
    anchor_match = anchor_pred.eq(targets) & (
        anchor_conf >= args.anchor_min_confidence
    )

    normalized_features = F.normalize(anchor_features.float().cpu(), dim=1)
    similarity = normalized_features.mm(prototypes.t())
    proto_top2 = torch.topk(similarity, k=2, dim=1)
    proto_pred = proto_top2.indices[:, 0]
    proto_margin = proto_top2.values[:, 0] - proto_top2.values[:, 1]
    proto_match = proto_pred.eq(targets.cpu()) & (
        proto_margin >= args.prototype_min_margin
    )

    candidate = (
        agreement
        & (stable_conf >= sample_threshold)
        & (margin >= args.min_margin)
        & (entropy <= args.max_entropy)
    ).cpu()

    if strict_anchor:
        candidate &= anchor_match.cpu() & proto_match
    else:
        candidate &= anchor_match.cpu() | proto_match

    targets_cpu = targets.cpu()
    sample_idx = sample_idx.long().cpu()
    same_label = bank_label[sample_idx].eq(targets_cpu)
    new_streak = torch.where(
        candidate,
        torch.where(
            same_label,
            bank_streak[sample_idx] + 1,
            torch.ones_like(bank_streak[sample_idx]),
        ),
        torch.zeros_like(bank_streak[sample_idx]),
    )

    bank_label[sample_idx] = torch.where(
        candidate,
        targets_cpu,
        torch.full_like(targets_cpu, -1),
    )
    bank_streak[sample_idx] = new_streak
    selected = candidate & (new_streak >= args.temporal_min_streak)

    conf_weight = (
        (stable_conf.cpu() - sample_threshold.cpu())
        / (1.0 - sample_threshold.cpu() + 1e-6)
    ).clamp(0.0, 1.0)
    margin_weight = (
        (margin.cpu() - args.min_margin)
        / (1.0 - args.min_margin + 1e-6)
    ).clamp(0.0, 1.0)
    entropy_weight = (
        (args.max_entropy - entropy.cpu()) / max(args.max_entropy, 1e-6)
    ).clamp(0.0, 1.0)
    proto_weight = (
        (proto_margin - args.prototype_min_margin) / 0.35
    ).clamp(0.0, 1.0)
    temporal_weight = (
        new_streak.float() / float(max(args.temporal_full_streak, 1))
    ).clamp(0.0, 1.0)

    quality = (
        0.35 * conf_weight
        + 0.20 * margin_weight
        + 0.15 * entropy_weight
        + 0.15 * anchor_conf.cpu().clamp(0.0, 1.0)
        + 0.15 * proto_weight
    )
    weight = quality * temporal_weight * selected.float()
    weight = weight * class_weight[targets_cpu]
    weight = weight.clamp(0.0, 1.5)

    align_mask = selected & (weight >= args.align_min_weight)

    return (
        targets_cpu,
        candidate,
        selected,
        weight,
        align_mask,
        margin.cpu(),
        entropy.cpu(),
        new_streak,
    )


def update_target_class_weights(selected_class):
    counts = torch.tensor(selected_class, dtype=torch.float32)
    valid = counts > 0
    weights = torch.ones(7, dtype=torch.float32)
    if valid.any():
        mean_count = counts[valid].mean()
        class_values = torch.sqrt(mean_count / counts[valid].clamp_min(1.0))
        class_values = class_values.clamp(0.75, 1.30)
        class_values[counts[valid] < 32] = torch.minimum(
            class_values[counts[valid] < 32],
            torch.ones_like(class_values[counts[valid] < 32]),
        )
        weights[valid] = class_values
    return weights


def scheduled_affinity_weight(epoch, args):
    if args.target_w2 <= 0 or epoch < args.affinity_warmup:
        return 0.0
    ramp = float(epoch - args.affinity_warmup + 1) / float(
        max(args.affinity_ramp, 1)
    )
    return args.target_w2 * min(1.0, ramp)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CAST RAF-DB -> FER2013 training, optimized for ResNet50"
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=["resnet18", "resnet50", "mobilenet_v2"],
    )
    parser.add_argument("-c", "--checkpoint", default=None)
    parser.add_argument(
        "--source_root",
        default="/workspace/ttt/code/test-upload-clean/datesets/raf-basic",
    )
    parser.add_argument(
        "--target_root",
        default="/workspace/ttt/code/data/fer2013",
    )
    parser.add_argument("--model_dir", default="./models/cast_resnet50")
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
    parser.add_argument("--target_lambda", type=float, default=0.50)
    parser.add_argument("--pseudo_ramp", type=int, default=6)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)

    parser.add_argument("--phi", type=float, default=1.4)
    parser.add_argument("--teacher_temperature", type=float, default=1.5)
    parser.add_argument("--pseudo_min_threshold", type=float, default=0.55)
    parser.add_argument("--pseudo_max_threshold", type=float, default=0.90)
    parser.add_argument("--threshold_quantile", type=float, default=0.60)
    parser.add_argument("--min_margin", type=float, default=0.08)
    parser.add_argument("--max_entropy", type=float, default=0.75)
    parser.add_argument("--anchor_min_confidence", type=float, default=0.40)
    parser.add_argument("--prototype_min_margin", type=float, default=0.01)
    parser.add_argument("--anchor_guard_epochs", type=int, default=7)
    parser.add_argument("--temporal_min_streak", type=int, default=2)
    parser.add_argument("--temporal_full_streak", type=int, default=3)

    parser.add_argument("--target_w2", type=float, default=0.10)
    parser.add_argument("--affinity_warmup", type=int, default=6)
    parser.add_argument("--affinity_ramp", type=int, default=6)
    parser.add_argument("--align_min_weight", type=float, default=0.35)
    parser.add_argument("--min_align_samples", type=int, default=8)

    args = parser.parse_args()

    if not 0.0 <= args.pseudo_min_threshold <= args.pseudo_max_threshold <= 0.99:
        parser.error(
            "Require 0 <= pseudo_min_threshold <= pseudo_max_threshold <= 0.99"
        )
    if not 0.0 <= args.threshold_quantile <= 0.90:
        parser.error("threshold_quantile must be in [0, 0.90]")
    if args.temporal_min_streak < 1:
        parser.error("temporal_min_streak must be >= 1")
    if args.temporal_full_streak < args.temporal_min_streak:
        parser.error("temporal_full_streak must be >= temporal_min_streak")
    if args.teacher_temperature <= 0:
        parser.error("teacher_temperature must be > 0")

    return args


def train_source(model, loader, source_weights, args, source_path):
    criterion = torch.nn.CrossEntropyLoss(
        weight=source_weights, reduction="none"
    )
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.source_lr_gamma
    )

    for epoch in range(args.source_epochs):
        model.train()
        correct = 0
        seen = 0
        cls_total = 0.0
        aff_total = 0.0
        steps = 0

        for imgs, targets in loader:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.long().cuda(non_blocking=True)

            output = model(
                imgs,
                targets,
                None,
                "train",
                "source",
                compute_affinity=args.w2 > 0,
            )
            cls_loss = criterion(output[0], targets).mean()
            aff_loss = output[1]
            weight_loss = classifier_weight_loss(model)
            loss = (
                args.w1 * cls_loss
                + args.w2 * aff_loss
                + args.w3 * weight_loss
            )

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite source loss")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output[0].argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            seen += targets.numel()
            cls_total += float(cls_loss.detach())
            aff_total += float(aff_loss.detach())
            steps += 1

        scheduler.step()
        print(
            "[Source %d] acc %.4f cls %.4f aff %.4f lr %.6f"
            % (
                epoch,
                correct / float(max(seen, 1)),
                cls_total / max(steps, 1),
                aff_total / max(steps, 1),
                optimizer.param_groups[0]["lr"],
            )
        )

    save_checkpoint(model, optimizer, source_path, args.source_epochs - 1, 0.0, "source")
    print("Source checkpoint saved:", source_path)
    return model


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    transforms_dict = make_transforms()
    source_train = RafDataSet(
        args.source_root,
        "train",
        transform=transforms_dict["source"],
        strong_transform=None,
        basic_aug=False,
    )
    source_proto = RafDataSet(
        args.source_root,
        "train",
        transform=transforms_dict["prototype"],
        strong_transform=None,
        basic_aug=False,
    )
    target_base = FER(
        args.target_root,
        "train",
        transform=transforms_dict["weak"],
        strong_transform=transforms_dict["strong"],
        basic_aug=False,
    )
    target_train = IndexedTargetDataset(target_base)
    target_test = FER(
        args.target_root,
        "test",
        transform=transforms_dict["test"],
        strong_transform=None,
    )

    if len(target_base) != 28709:
        print(
            "WARNING: FER2013 train contains %d images; the paper uses 28709."
            % len(target_base)
        )
    if len(target_test) != 3589:
        print(
            "WARNING: FER2013 test contains %d images; the paper uses 3589."
            % len(target_test)
        )

    source_loader = make_loader(
        source_train,
        args.batch_size,
        args.workers,
        True,
        True,
        1,
    )
    proto_loader = make_loader(
        source_proto,
        args.batch_size,
        args.workers,
        False,
        False,
        2,
    )
    target_loader = make_loader(
        target_train,
        args.batch_size,
        args.workers,
        True,
        True,
        3,
    )
    threshold_loader = make_loader(
        target_train,
        args.batch_size,
        args.workers,
        False,
        False,
        4,
    )
    test_loader = make_loader(
        target_test,
        args.batch_size,
        args.workers,
        False,
        False,
        5,
    )

    model = Networks.Model(backbone=args.backbone, num_classes=7).cuda()
    source_weights = source_class_weights(source_train, torch.device("cuda"))
    print(
        "Source class weights:",
        [round(float(x), 4) for x in source_weights.detach().cpu().tolist()],
    )

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
        model = train_source(
            model,
            source_loader,
            source_weights,
            args,
            source_path,
        )

    anchor = copy.deepcopy(model).cuda().eval()
    teacher = copy.deepcopy(model).cuda().eval()
    for network in (anchor, teacher):
        for parameter in network.parameters():
            parameter.requires_grad = False

    prototypes, proto_counts = build_source_prototypes(anchor, proto_loader)
    print("Source prototype counts:", proto_counts.tolist())

    feature_lr = args.target_lr * (0.20 if args.backbone == "resnet50" else 0.35)
    optimizer = torch.optim.Adam(
        [
            {"params": model.feature.parameters(), "lr": feature_lr},
            {"params": model.fc.parameters(), "lr": args.target_lr},
            {"params": model.bn.parameters(), "lr": args.target_lr},
        ],
        weight_decay=1e-4,
    )
    target_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.target_lr_gamma
    )
    source_criterion = torch.nn.CrossEntropyLoss(
        weight=source_weights, reduction="none"
    )
    target_criterion = torch.nn.CrossEntropyLoss(reduction="none")

    bank_label = torch.full((len(target_train),), -1, dtype=torch.long)
    bank_streak = torch.zeros(len(target_train), dtype=torch.long)
    target_class_weight = torch.ones(7, dtype=torch.float32)

    initial_acc = evaluate(model, test_loader, len(target_test))
    print("[Target start] accuracy %.4f" % initial_acc)
    best_student = initial_acc
    best_ema = initial_acc
    best_overall = initial_acc
    save_checkpoint(model, optimizer, student_best_path, -1, initial_acc, "student")
    save_checkpoint(teacher, None, ema_best_path, -1, initial_acc, "ema")
    save_checkpoint(model, optimizer, best_path, -1, initial_acc, "source")

    source_iter = iter(source_loader)

    for epoch in range(args.epochs):
        freeze_backbone = epoch < args.freeze_backbone_epochs
        for parameter in model.feature.parameters():
            parameter.requires_grad = not freeze_backbone

        strict_anchor = epoch < args.anchor_guard_epochs
        (
            thresholds,
            threshold_support,
            predicted_count,
            weak_agreement,
            anchor_agreement,
        ) = estimate_target_thresholds(
            teacher,
            anchor,
            threshold_loader,
            args,
            epoch,
        )

        pseudo_scale = min(
            1.0,
            float(epoch + 1) / float(max(args.pseudo_ramp, 1)),
        )
        scheduled_w2 = scheduled_affinity_weight(epoch, args)

        print(
            "[Epoch %d] CATM %s support %s predicted %s weak_agree %.4f "
            "anchor_agree %.4f guard %s backbone %s w2 %.4f pseudo_scale %.3f"
            % (
                epoch,
                [round(float(x), 4) for x in thresholds.tolist()],
                threshold_support.tolist(),
                predicted_count.tolist(),
                weak_agreement,
                anchor_agreement,
                "strict" if strict_anchor else "relaxed",
                "frozen" if freeze_backbone else "train",
                scheduled_w2,
                pseudo_scale,
            )
        )

        candidate_total = 0
        selected_total = 0
        align_total = 0
        pseudo_correct = 0
        pseudo_total = 0
        candidate_class = np.zeros(7, dtype=np.int64)
        selected_class = np.zeros(7, dtype=np.int64)
        class_correct = np.zeros(7, dtype=np.int64)
        weight_sum = 0.0
        margin_sum = 0.0
        entropy_sum = 0.0
        streak_sum = 0.0
        source_ce_sum = 0.0
        target_ce_sum = 0.0
        applied_w2_sum = 0.0
        steps = 0

        for imgs_w1, imgs_w2, imgs_strong, gt_target, sample_idx in target_loader:
            try:
                source_imgs, source_targets = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_imgs, source_targets = next(source_iter)

            imgs_w1_cuda = imgs_w1.cuda(non_blocking=True)
            imgs_w2_cuda = imgs_w2.cuda(non_blocking=True)

            with torch.no_grad():
                teacher_out_1, _ = teacher(
                    imgs_w1_cuda, None, None, "test", "target"
                )
                teacher_out_2, _ = teacher(
                    imgs_w2_cuda, None, None, "test", "target"
                )
                anchor_out, anchor_features = anchor(
                    imgs_w1_cuda, None, None, "test", "target"
                )

                (
                    pseudo_targets,
                    candidate_mask,
                    selected_mask,
                    pseudo_weight,
                    align_mask,
                    margin,
                    entropy,
                    streak,
                ) = select_pseudo_labels(
                    teacher_out_1,
                    teacher_out_2,
                    anchor_out,
                    anchor_features,
                    prototypes,
                    thresholds,
                    sample_idx,
                    bank_label,
                    bank_streak,
                    target_class_weight,
                    args,
                    strict_anchor,
                )

            candidate_n = int(candidate_mask.sum().item())
            selected_n = int(selected_mask.sum().item())
            align_n = int(align_mask.sum().item())
            candidate_total += candidate_n
            selected_total += selected_n
            align_total += align_n

            gt_cpu = gt_target.long().cpu()
            for c in range(7):
                candidate_class[c] += int(
                    (candidate_mask & pseudo_targets.eq(c)).sum().item()
                )

            if selected_mask.any():
                correct = pseudo_targets[selected_mask].eq(gt_cpu[selected_mask])
                pseudo_correct += int(correct.sum().item())
                pseudo_total += selected_n
                weight_sum += float(pseudo_weight[selected_mask].sum().item())
                margin_sum += float(margin[selected_mask].sum().item())
                entropy_sum += float(entropy[selected_mask].sum().item())
                streak_sum += float(streak[selected_mask].float().sum().item())

                for c in range(7):
                    class_mask = selected_mask & pseudo_targets.eq(c)
                    n_class = int(class_mask.sum().item())
                    if n_class > 0:
                        selected_class[c] += n_class
                        class_correct[c] += int(
                            pseudo_targets[class_mask]
                            .eq(gt_cpu[class_mask])
                            .sum()
                            .item()
                        )

            model.train()
            if freeze_backbone:
                model.feature.eval()

            source_imgs = source_imgs.cuda(non_blocking=True)
            source_targets = source_targets.long().cuda(non_blocking=True)
            imgs_strong = imgs_strong.cuda(non_blocking=True)
            pseudo_targets_cuda = pseudo_targets.long().cuda(non_blocking=True)
            selected_cuda = selected_mask.cuda(non_blocking=True)
            pseudo_weight_cuda = pseudo_weight.cuda(non_blocking=True)
            align_cuda = align_mask.cuda(non_blocking=True)

            n_source = source_imgs.shape[0]
            train_imgs = torch.cat((source_imgs, imgs_strong), dim=0)
            train_targets = torch.cat(
                (source_targets, pseudo_targets_cuda), dim=0
            )
            affinity_mask = torch.cat(
                (
                    torch.ones(
                        n_source, dtype=torch.bool, device=source_imgs.device
                    ),
                    align_cuda,
                ),
                dim=0,
            )

            applied_w2 = (
                scheduled_w2
                if align_n >= args.min_align_samples
                else 0.0
            )
            output = model(
                train_imgs,
                train_targets,
                affinity_mask,
                "train",
                "target",
                source_count=n_source,
                compute_affinity=applied_w2 > 0,
            )

            source_loss = source_criterion(
                output[0][:n_source], source_targets
            ).mean()

            if selected_n > 0:
                per_target_loss = target_criterion(
                    output[0][n_source:], pseudo_targets_cuda
                )
                effective_weight = (
                    pseudo_weight_cuda * selected_cuda.float()
                )
                target_loss = (
                    per_target_loss * effective_weight
                ).sum() / effective_weight.sum().clamp_min(1.0)
            else:
                target_loss = source_loss.new_zeros(())

            classification_loss = (
                source_loss
                + args.target_lambda * pseudo_scale * target_loss
            )
            loss = (
                args.w1 * classification_loss
                + applied_w2 * output[1]
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
            applied_w2_sum += applied_w2
            steps += 1

        target_scheduler.step()
        target_class_weight = update_target_class_weights(selected_class)

        pseudo_acc = pseudo_correct / float(max(pseudo_total, 1))
        class_acc = [
            round(
                class_correct[c] / float(selected_class[c]),
                4,
            )
            if selected_class[c] > 0
            else 0.0
            for c in range(7)
        ]

        print(
            "[Epoch %d] candidate %d selected %d align %d pseudo_acc %.4f "
            "class_acc %s candidate_class %s selected_class %s class_weight %s "
            "mean_weight %.4f margin %.4f entropy %.4f streak %.2f "
            "source_ce %.4f target_ce %.4f applied_w2 %.4f"
            % (
                epoch,
                candidate_total,
                selected_total,
                align_total,
                pseudo_acc,
                class_acc,
                candidate_class.tolist(),
                selected_class.tolist(),
                [round(float(x), 3) for x in target_class_weight.tolist()],
                weight_sum / max(pseudo_total, 1),
                margin_sum / max(pseudo_total, 1),
                entropy_sum / max(pseudo_total, 1),
                streak_sum / max(pseudo_total, 1),
                source_ce_sum / max(steps, 1),
                target_ce_sum / max(steps, 1),
                applied_w2_sum / max(steps, 1),
            )
        )

        student_acc = evaluate(model, test_loader, len(target_test))
        ema_acc = evaluate(teacher, test_loader, len(target_test))
        print(
            "[Epoch %d] Student accuracy: %.4f | EMA accuracy: %.4f"
            % (epoch, student_acc, ema_acc)
        )

        if student_acc > best_student:
            best_student = student_acc
            save_checkpoint(
                model,
                optimizer,
                student_best_path,
                epoch,
                student_acc,
                "student",
            )

        if ema_acc > best_ema:
            best_ema = ema_acc
            save_checkpoint(
                teacher,
                None,
                ema_best_path,
                epoch,
                ema_acc,
                "ema",
            )

        epoch_best = max(student_acc, ema_acc)
        if epoch_best > best_overall:
            if ema_acc >= student_acc:
                best_overall = ema_acc
                save_checkpoint(
                    teacher,
                    None,
                    best_path,
                    epoch,
                    ema_acc,
                    "ema",
                )
            else:
                best_overall = student_acc
                save_checkpoint(
                    model,
                    optimizer,
                    best_path,
                    epoch,
                    student_acc,
                    "student",
                )
            print("Best checkpoint:", best_path, "acc", best_overall)

    final_student_acc = evaluate(model, test_loader, len(target_test))
    final_ema_acc = evaluate(teacher, test_loader, len(target_test))
    save_checkpoint(
        model,
        optimizer,
        student_final_path,
        args.epochs - 1,
        final_student_acc,
        "student_final",
    )
    save_checkpoint(
        teacher,
        None,
        ema_final_path,
        args.epochs - 1,
        final_ema_acc,
        "ema_final",
    )

    print(
        "best_student %.4f best_ema %.4f best_overall %.4f final_student %.4f final_ema %.4f"
        % (
            best_student,
            best_ema,
            best_overall,
            final_student_acc,
            final_ema_acc,
        )
    )


if __name__ == "__main__":
    main()
