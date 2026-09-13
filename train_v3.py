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
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class IndexedTargetDataset(Dataset):
    """Add a stable sample id without changing dataset.py's public API."""

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
    g = torch.Generator()
    g.manual_seed(SEED + seed_offset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
    )


def make_transforms():
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return {
        "source": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32),
            ], p=0.5),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ]),
        "weak": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]),
        "strong": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32),
            ], p=0.5),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ]),
        "test": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm,
        ]),
    }


def classifier_weight_loss(model):
    w = F.normalize(model.fc.weight, dim=1)
    eye = torch.eye(w.shape[0], device=w.device, dtype=w.dtype)
    return ((w.mm(w.t()) - eye + 1.0) / 2.0).mean()


@torch.no_grad()
def update_ema(student, teacher, decay):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        if tb.dtype.is_floating_point:
            tb.data.mul_(decay).add_(sb.data, alpha=1.0 - decay)
        else:
            tb.data.copy_(sb.data)


@torch.no_grad()
def evaluate(model, loader, num):
    model.eval()
    correct = 0
    preds, labels = [], []
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
                sums[c] += features[mask].sum(0)
                counts[c] += int(mask.sum())
    if sums is None or (counts == 0).any():
        raise RuntimeError("Cannot build complete source prototypes: %s" % counts.tolist())
    prototypes = F.normalize(sums / counts.float().unsqueeze(1), dim=1)
    return prototypes, counts


@torch.no_grad()
def estimate_thresholds(teacher, anchor, loader, args, epoch):
    class_all = [[] for _ in range(7)]
    class_stable = [[] for _ in range(7)]
    predicted = torch.zeros(7, dtype=torch.long)
    prior_sum = torch.zeros(7, dtype=torch.float64)
    total = 0
    weak_ok = 0
    anchor_ok = 0

    teacher.eval()
    anchor.eval()
    for w1, w2, _, _, _ in loader:
        w1 = w1.cuda(non_blocking=True)
        w2 = w2.cuda(non_blocking=True)
        t1, _ = teacher(w1, None, None, "test", "target")
        t2, _ = teacher(w2, None, None, "test", "target")
        ao, _ = anchor(w1, None, None, "test", "target")

        p1 = F.softmax(t1, dim=1).cpu()
        p2 = F.softmax(t2, dim=1).cpu()
        ap = F.softmax(ao, dim=1).cpu()
        avg = 0.5 * (p1 + p2)
        conf, target = avg.max(1)
        weak = p1.argmax(1).eq(p2.argmax(1))
        anchor_conf, anchor_pred = ap.max(1)
        anchored = anchor_pred.eq(target) & (anchor_conf >= args.anchor_min_confidence)
        stable_conf = torch.minimum(
            p1.gather(1, target[:, None]).squeeze(1),
            p2.gather(1, target[:, None]).squeeze(1),
        )

        prior_sum += avg.double().sum(0)
        total += avg.shape[0]
        weak_ok += int(weak.sum())
        anchor_ok += int((weak & anchored).sum())
        predicted += torch.bincount(target, minlength=7)
        for c in range(7):
            v = conf[target == c]
            if v.numel():
                class_all[c].append(v)
            v = stable_conf[(target == c) & weak & anchored]
            if v.numel():
                class_stable[c].append(v)

    progress = epoch / float(max(args.epochs - 1, 1))
    q = min(0.95, args.threshold_quantile + 0.05 * progress)
    stage = args.epochs / float(max(args.epochs - epoch, 1))
    threshold = torch.full((7,), args.pseudo_max_threshold)
    support = torch.zeros(7, dtype=torch.long)

    for c in range(7):
        if not class_all[c]:
            continue
        all_v = torch.cat(class_all[c])
        cast_t = float(all_v.mean()) * args.phi * stage
        cast_t = float(np.clip(cast_t, args.pseudo_min_threshold, args.pseudo_max_threshold))
        if class_stable[c]:
            stable_v = torch.cat(class_stable[c])
            support[c] = stable_v.numel()
            if stable_v.numel() >= 8:
                robust_t = float(torch.quantile(stable_v, q))
                cast_t = max(cast_t, robust_t)
            else:
                cast_t = args.pseudo_max_threshold
        else:
            cast_t = args.pseudo_max_threshold
        threshold[c] = float(np.clip(cast_t, args.pseudo_min_threshold, args.pseudo_max_threshold))

    prior = (prior_sum / max(total, 1)).float()
    prior /= prior.sum().clamp_min(1e-12)
    return threshold, prior, support, predicted, weak_ok / max(total, 1), anchor_ok / max(total, 1)


@torch.no_grad()
def select_pseudo(
    t1,
    t2,
    anchor_out,
    anchor_features,
    prototypes,
    thresholds,
    sample_idx,
    bank_label,
    bank_streak,
    args,
    strict_anchor,
):
    p1 = F.softmax(t1, dim=1)
    p2 = F.softmax(t2, dim=1)
    avg = 0.5 * (p1 + p2)
    target = avg.argmax(1)
    weak = p1.argmax(1).eq(p2.argmax(1))
    stable_conf = torch.minimum(
        p1.gather(1, target[:, None]).squeeze(1),
        p2.gather(1, target[:, None]).squeeze(1),
    )
    sample_threshold = thresholds.to(t1.device)[target]
    top2 = torch.topk(avg, 2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(avg.clamp_min(1e-8) * avg.clamp_min(1e-8).log()).sum(1)
    entropy /= np.log(7.0)

    ap = F.softmax(anchor_out, dim=1)
    anchor_conf, anchor_pred = ap.max(1)
    anchor_match = anchor_pred.eq(target) & (anchor_conf >= args.anchor_min_confidence)

    feature = F.normalize(anchor_features.float().cpu(), dim=1)
    similarity = feature.mm(prototypes.t())
    proto = torch.topk(similarity, 2, dim=1)
    proto_pred = proto.indices[:, 0]
    proto_margin = proto.values[:, 0] - proto.values[:, 1]
    proto_match = proto_pred.eq(target.cpu()) & (proto_margin >= args.prototype_min_margin)

    candidate = (
        weak
        & (stable_conf >= sample_threshold)
        & (margin >= args.min_margin)
        & (entropy <= args.max_entropy)
    ).cpu()
    if strict_anchor:
        candidate &= anchor_match.cpu() & proto_match
    else:
        candidate &= anchor_match.cpu() | proto_match

    target_cpu = target.cpu()
    idx = sample_idx.long().cpu()
    same = bank_label[idx].eq(target_cpu)
    new_streak = torch.where(
        candidate,
        torch.where(same, bank_streak[idx] + 1, torch.ones_like(bank_streak[idx])),
        torch.zeros_like(bank_streak[idx]),
    )
    bank_label[idx] = torch.where(candidate, target_cpu, torch.full_like(target_cpu, -1))
    bank_streak[idx] = new_streak
    selected = candidate & (new_streak >= args.temporal_min_streak)

    conf_w = ((stable_conf.cpu() - sample_threshold.cpu()) / (1.0 - sample_threshold.cpu() + 1e-6)).clamp(0, 1)
    margin_w = ((margin.cpu() - args.min_margin) / (1.0 - args.min_margin + 1e-6)).clamp(0, 1)
    entropy_w = ((args.max_entropy - entropy.cpu()) / max(args.max_entropy, 1e-6)).clamp(0, 1)
    anchor_w = anchor_conf.cpu().clamp(0, 1)
    proto_w = ((proto_margin - args.prototype_min_margin) / 0.5).clamp(0, 1)
    temporal_w = (new_streak.float() / max(args.temporal_full_streak, 1)).clamp(0, 1)
    weight = torch.sqrt(conf_w * margin_w * entropy_w * anchor_w * proto_w)
    weight = weight * temporal_w * selected.float()
    align = selected & (weight >= args.align_min_weight)
    return target_cpu, candidate, selected, weight, align, margin.cpu(), entropy.cpu(), new_streak


def parse_args():
    p = argparse.ArgumentParser(description="Conservative CAST target adaptation")
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v2"])
    p.add_argument("--source_root", default="/workspace/ttt/code/test-upload-clean/datesets/raf-basic")
    p.add_argument("--target_root", default="/workspace/ttt/code/data/fer2013")
    p.add_argument("--model_dir", default="./models/cast_fix_v3")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--target_lr", type=float, default=5e-5)
    p.add_argument("--target_lr_gamma", type=float, default=0.97)
    p.add_argument("--ema_decay", type=float, default=0.9995)
    p.add_argument("--w1", type=float, default=4.0)
    p.add_argument("--w3", type=float, default=0.1)
    p.add_argument("--target_lambda", type=float, default=0.25)
    p.add_argument("--phi", type=float, default=1.4)
    p.add_argument("--pseudo_min_threshold", type=float, default=0.75)
    p.add_argument("--pseudo_max_threshold", type=float, default=0.95)
    p.add_argument("--threshold_quantile", type=float, default=0.65)
    p.add_argument("--min_margin", type=float, default=0.20)
    p.add_argument("--max_entropy", type=float, default=0.60)
    p.add_argument("--anchor_min_confidence", type=float, default=0.55)
    p.add_argument("--prototype_min_margin", type=float, default=0.02)
    p.add_argument("--temporal_min_streak", type=int, default=2)
    p.add_argument("--temporal_full_streak", type=int, default=3)
    p.add_argument("--pseudo_ramp", type=int, default=10)
    p.add_argument("--freeze_backbone_epochs", type=int, default=3)
    p.add_argument("--anchor_guard_epochs", type=int, default=12)
    p.add_argument("--target_w2", type=float, default=0.0)
    p.add_argument("--affinity_warmup", type=int, default=12)
    p.add_argument("--affinity_ramp", type=int, default=5)
    p.add_argument("--align_min_weight", type=float, default=0.40)
    p.add_argument("--min_align_samples", type=int, default=8)
    a = p.parse_args()
    if not 0 <= a.pseudo_min_threshold <= a.pseudo_max_threshold <= 0.99:
        p.error("invalid pseudo threshold range")
    if a.temporal_min_streak < 1 or a.temporal_full_streak < a.temporal_min_streak:
        p.error("invalid temporal streak configuration")
    return a


def affinity_weight(epoch, args):
    if args.target_w2 <= 0 or epoch < args.affinity_warmup:
        return 0.0
    x = (epoch - args.affinity_warmup + 1) / float(max(args.affinity_ramp, 1))
    return args.target_w2 * min(1.0, x)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    tx = make_transforms()

    source_train = RafDataSet(args.source_root, "train", transform=tx["source"], strong_transform=None)
    source_proto = RafDataSet(args.source_root, "train", transform=tx["test"], strong_transform=None)
    target_base = FER(args.target_root, "train", transform=tx["weak"], strong_transform=tx["strong"])
    target_train = IndexedTargetDataset(target_base)
    target_test = FER(args.target_root, "test", transform=tx["test"], strong_transform=None)

    if len(target_base) != 28709 or len(target_test) != 3589:
        print("WARNING: FER split differs from paper: train=%d test=%d" % (len(target_base), len(target_test)))

    source_loader = make_loader(source_train, args.batch_size, args.workers, True, True, 1)
    proto_loader = make_loader(source_proto, args.batch_size, args.workers, False, False, 2)
    target_loader = make_loader(target_train, args.batch_size, args.workers, True, True, 3)
    threshold_loader = make_loader(target_train, args.batch_size, args.workers, False, False, 4)
    test_batch = 100 if args.backbone == "resnet50" else args.batch_size
    test_loader = make_loader(target_test, test_batch, args.workers, False, False, 5)

    model = Networks.Model(backbone=args.backbone, num_classes=7).cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(checkpoint["model"], strict=True)
    print("Loaded source checkpoint:", args.checkpoint)

    anchor = copy.deepcopy(model).cuda().eval()
    teacher = copy.deepcopy(model).cuda().eval()
    for net in (anchor, teacher):
        for p in net.parameters():
            p.requires_grad = False

    prototypes, proto_count = build_source_prototypes(anchor, proto_loader)
    print("Source prototype counts:", proto_count.tolist())

    optimizer = torch.optim.Adam(model.parameters(), args.target_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.target_lr_gamma)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    bank_label = torch.full((len(target_train),), -1, dtype=torch.long)
    bank_streak = torch.zeros(len(target_train), dtype=torch.long)

    source_acc = evaluate(model, test_loader, len(target_test))
    print("[Epoch -1] Source/Student accuracy: %.4f" % source_acc)
    best_student = source_acc
    best_ema = source_acc
    best_overall = source_acc
    student_path = os.path.join(args.model_dir, "resnet18_rafdb_fer_student_best.pth")
    ema_path = os.path.join(args.model_dir, "resnet18_rafdb_fer_ema_best.pth")
    best_path = os.path.join(args.model_dir, "resnet18_rafdb_fer_best.pth")
    anchor_path = os.path.join(args.model_dir, "resnet18_rafdb_fer_source_anchor.pth")
    save_checkpoint(anchor, None, anchor_path, -1, source_acc, "source_anchor")
    save_checkpoint(model, optimizer, student_path, -1, source_acc, "student")
    save_checkpoint(teacher, None, ema_path, -1, source_acc, "ema")
    save_checkpoint(model, optimizer, best_path, -1, source_acc, "source")

    source_iter = iter(source_loader)
    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_backbone_epochs
        for p in model.feature.parameters():
            p.requires_grad = not frozen
        strict_anchor = epoch < args.anchor_guard_epochs
        thresholds, _, support, predicted, weak_rate, anchor_rate = estimate_thresholds(
            teacher, anchor, threshold_loader, args, epoch
        )
        pseudo_scale = min(1.0, (epoch + 1) / float(max(args.pseudo_ramp, 1)))
        scheduled_w2 = affinity_weight(epoch, args)
        print(
            "[Epoch %d] CATM %s support %s predicted %s weak_agree %.4f anchor_agree %.4f "
            "guard %s backbone %s w2 %.4f pseudo_scale %.3f"
            % (
                epoch,
                [round(float(x), 4) for x in thresholds.tolist()],
                support.tolist(),
                predicted.tolist(),
                weak_rate,
                anchor_rate,
                "strict" if strict_anchor else "relaxed",
                "frozen" if frozen else "train",
                scheduled_w2,
                pseudo_scale,
            )
        )

        candidate_total = selected_total = align_total = pseudo_correct = 0
        pseudo_total = 0
        class_total = np.zeros(7, dtype=np.int64)
        class_correct = np.zeros(7, dtype=np.int64)
        candidate_class = np.zeros(7, dtype=np.int64)
        weight_sum = margin_sum = entropy_sum = streak_sum = 0.0
        source_ce_sum = target_ce_sum = applied_sum = 0.0
        steps = 0

        for w1, w2, strong, gt, sample_idx in target_loader:
            try:
                src_img, src_y = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                src_img, src_y = next(source_iter)

            w1_cuda = w1.cuda(non_blocking=True)
            w2_cuda = w2.cuda(non_blocking=True)
            with torch.no_grad():
                t1, _ = teacher(w1_cuda, None, None, "test", "target")
                t2, _ = teacher(w2_cuda, None, None, "test", "target")
                ao, af = anchor(w1_cuda, None, None, "test", "target")
                target, candidate, selected, weight, align, margin, entropy, streak = select_pseudo(
                    t1, t2, ao, af, prototypes, thresholds, sample_idx,
                    bank_label, bank_streak, args, strict_anchor
                )

            candidate_n = int(candidate.sum())
            selected_n = int(selected.sum())
            align_n = int(align.sum())
            candidate_total += candidate_n
            selected_total += selected_n
            align_total += align_n
            gt = gt.long().cpu()
            for c in range(7):
                candidate_class[c] += int((candidate & target.eq(c)).sum())
            if selected.any():
                pseudo_correct += int(target[selected].eq(gt[selected]).sum())
                pseudo_total += selected_n
                weight_sum += float(weight[selected].sum())
                margin_sum += float(margin[selected].sum())
                entropy_sum += float(entropy[selected].sum())
                streak_sum += float(streak[selected].float().sum())
                for c in range(7):
                    m = selected & target.eq(c)
                    n = int(m.sum())
                    if n:
                        class_total[c] += n
                        class_correct[c] += int(target[m].eq(gt[m]).sum())

            model.train()
            if frozen:
                model.feature.eval()
            src_img = src_img.cuda(non_blocking=True)
            src_y = src_y.long().cuda(non_blocking=True)
            strong = strong.cuda(non_blocking=True)
            target_cuda = target.long().cuda(non_blocking=True)
            selected_cuda = selected.cuda(non_blocking=True)
            weight_cuda = weight.cuda(non_blocking=True)
            align_cuda = align.cuda(non_blocking=True)
            n_source = src_img.shape[0]
            all_img = torch.cat((src_img, strong), 0)
            all_y = torch.cat((src_y, target_cuda), 0)
            con_idx = torch.cat((torch.ones(n_source, dtype=torch.bool, device="cuda"), align_cuda), 0)
            applied_w2 = scheduled_w2 if align_n >= args.min_align_samples else 0.0
            out = model(
                all_img, all_y, con_idx, "train", "target",
                source_count=n_source, compute_affinity=applied_w2 > 0,
            )
            source_ce = criterion(out[0][:n_source], src_y).mean()
            if selected_n:
                target_loss = criterion(out[0][n_source:], target_cuda)
                tw = weight_cuda * selected_cuda.float()
                target_ce = (target_loss * tw).sum() / tw.sum().clamp_min(1.0)
            else:
                target_ce = source_ce.new_zeros(())
            cls = source_ce + args.target_lambda * pseudo_scale * target_ce
            loss = cls * args.w1 + out[1] * applied_w2 + classifier_weight_loss(model) * args.w3
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite target loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(model, teacher, args.ema_decay)

            source_ce_sum += float(source_ce.detach())
            target_ce_sum += float(target_ce.detach())
            applied_sum += applied_w2
            steps += 1

        scheduler.step()
        pseudo_acc = pseudo_correct / float(max(pseudo_total, 1))
        class_acc = [
            round(class_correct[c] / float(class_total[c]), 4) if class_total[c] else 0.0
            for c in range(7)
        ]
        print(
            "[Epoch %d] candidate %d selected %d align %d pseudo_acc %.4f class_acc %s "
            "candidate_class %s selected_class %s mean_weight %.4f margin %.4f entropy %.4f "
            "streak %.2f source_ce %.4f target_ce %.4f applied_w2 %.4f"
            % (
                epoch, candidate_total, selected_total, align_total, pseudo_acc, class_acc,
                candidate_class.tolist(), class_total.tolist(),
                weight_sum / max(pseudo_total, 1), margin_sum / max(pseudo_total, 1),
                entropy_sum / max(pseudo_total, 1), streak_sum / max(pseudo_total, 1),
                source_ce_sum / max(steps, 1), target_ce_sum / max(steps, 1),
                applied_sum / max(steps, 1),
            )
        )

        student_acc = evaluate(model, test_loader, len(target_test))
        ema_acc = evaluate(teacher, test_loader, len(target_test))
        print("[Epoch %d] Student accuracy: %.4f | EMA accuracy: %.4f" % (epoch, student_acc, ema_acc))
        if student_acc > best_student:
            best_student = student_acc
            save_checkpoint(model, optimizer, student_path, epoch, student_acc, "student")
        if ema_acc > best_ema:
            best_ema = ema_acc
            save_checkpoint(teacher, None, ema_path, epoch, ema_acc, "ema")
        if max(student_acc, ema_acc) > best_overall:
            if ema_acc >= student_acc:
                best_overall = ema_acc
                save_checkpoint(teacher, None, best_path, epoch, ema_acc, "ema")
            else:
                best_overall = student_acc
                save_checkpoint(model, optimizer, best_path, epoch, student_acc, "student")
            print("Best overall checkpoint:", best_path, "acc", best_overall)

    print("best_student %.4f best_ema %.4f best_overall %.4f" % (best_student, best_ema, best_overall))


if __name__ == "__main__":
    main()
