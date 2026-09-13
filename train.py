import argparse
import copy
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def classifier_weight_loss(model):
    normalized = F.normalize(model.fc.weight, dim=1)
    identity = torch.eye(normalized.shape[0], device=normalized.device, dtype=normalized.dtype)
    return ((normalized.mm(normalized.t()) - identity + 1.0) / 2.0).mean()


@torch.no_grad()
def update_ema(student, teacher, decay):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if teacher_buffer.dtype.is_floating_point:
            teacher_buffer.data.mul_(decay).add_(student_buffer.data, alpha=1.0 - decay)
        else:
            teacher_buffer.data.copy_(student_buffer.data)


def adjust_probabilities(prob, prior, balance_power):
    if balance_power <= 0:
        return prob
    prior = prior.to(prob.device).clamp_min(1e-6)
    uniform = torch.full_like(prior, 1.0 / float(prior.numel()))
    correction = (uniform / prior).pow(balance_power).clamp(0.5, 2.0)
    adjusted = prob * correction.unsqueeze(0)
    return adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def estimate_target_statistics(
    teacher,
    loader,
    class_num,
    epoch,
    total_epochs,
    min_threshold,
    max_threshold,
    phi,
    threshold_quantile,
):
    """Estimate robust class-wise pseudo-label thresholds on the full target set.

    The CAST mean-confidence term is retained, but we additionally require a
    class-wise confidence quantile from samples on which two weak teacher views
    agree. This prevents a low-confidence tail class from getting an extremely
    permissive threshold and feeding noisy pseudo labels back into itself.
    """
    teacher.eval()
    prior_sum = torch.zeros(class_num, dtype=torch.float64)
    sample_count = 0
    agreement_num = 0
    agreement_total = 0
    class_confidence = [[] for _ in range(class_num)]
    class_all_confidence = [[] for _ in range(class_num)]
    predicted_count = torch.zeros(class_num, dtype=torch.long)

    for imgs_w1, imgs_w2, _, _ in loader:
        out_w1, _ = teacher(imgs_w1.cuda(non_blocking=True), None, None, "test", "target")
        out_w2, _ = teacher(imgs_w2.cuda(non_blocking=True), None, None, "test", "target")
        prob_1 = F.softmax(out_w1, dim=1).cpu()
        prob_2 = F.softmax(out_w2, dim=1).cpu()
        avg_prob = 0.5 * (prob_1 + prob_2)

        confidence, targets = avg_prob.max(dim=1)
        pred_1 = prob_1.argmax(dim=1)
        pred_2 = prob_2.argmax(dim=1)
        agreement = pred_1.eq(pred_2)

        target_conf_1 = prob_1.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_conf_2 = prob_2.gather(1, targets.unsqueeze(1)).squeeze(1)
        stable_conf = torch.minimum(target_conf_1, target_conf_2)

        prior_sum += avg_prob.double().sum(dim=0)
        sample_count += avg_prob.shape[0]
        agreement_num += int(agreement.sum().item())
        agreement_total += int(agreement.numel())
        predicted_count += torch.bincount(targets, minlength=class_num)

        for c in range(class_num):
            all_values = confidence[targets == c]
            if all_values.numel():
                class_all_confidence[c].append(all_values)
            stable_values = stable_conf[(targets == c) & agreement]
            if stable_values.numel():
                class_confidence[c].append(stable_values)

    progress = float(epoch) / float(max(total_epochs - 1, 1))
    q = float(np.clip(threshold_quantile + 0.10 * progress, 0.0, 0.95))
    stage_factor = float(total_epochs) / float(max(total_epochs - epoch, 1))
    thresholds = torch.full((class_num,), float(max_threshold), dtype=torch.float32)
    support = torch.zeros(class_num, dtype=torch.long)

    for c in range(class_num):
        if not class_all_confidence[c]:
            continue

        all_values = torch.cat(class_all_confidence[c])
        cast_pc = float(all_values.mean().item()) * float(phi) * stage_factor
        cast_pc = float(np.clip(cast_pc, min_threshold, max_threshold))

        if class_confidence[c]:
            stable_values = torch.cat(class_confidence[c])
            support[c] = stable_values.numel()
            if stable_values.numel() >= 8:
                quantile_threshold = float(torch.quantile(stable_values, q).item())
                threshold = max(cast_pc, quantile_threshold, float(min_threshold))
            else:
                threshold = float(max_threshold)
        else:
            threshold = float(max_threshold)

        thresholds[c] = float(np.clip(threshold, min_threshold, max_threshold))

    prior = (prior_sum / max(sample_count, 1)).float()
    prior = prior / prior.sum().clamp_min(1e-12)
    agreement_rate = agreement_num / max(agreement_total, 1)
    return thresholds, prior, support, agreement_rate, predicted_count


@torch.no_grad()
def build_pseudo_labels(
    out_w1,
    out_w2,
    thresholds,
    prior,
    balance_power,
    min_margin,
    max_entropy,
    align_min_weight,
):
    prob_1 = adjust_probabilities(F.softmax(out_w1, dim=1), prior, balance_power)
    prob_2 = adjust_probabilities(F.softmax(out_w2, dim=1), prior, balance_power)
    avg_prob = 0.5 * (prob_1 + prob_2)

    pred_1 = prob_1.argmax(dim=1)
    pred_2 = prob_2.argmax(dim=1)
    targets = avg_prob.argmax(dim=1)
    agreement = pred_1.eq(pred_2)

    target_conf_1 = prob_1.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_conf_2 = prob_2.gather(1, targets.unsqueeze(1)).squeeze(1)
    stable_conf = torch.minimum(target_conf_1, target_conf_2)
    sample_threshold = thresholds.to(out_w1.device)[targets]

    top2 = torch.topk(avg_prob, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(avg_prob.clamp_min(1e-8) * avg_prob.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / np.log(float(avg_prob.shape[1]))

    selected = (
        agreement
        & (stable_conf >= sample_threshold)
        & (margin >= float(min_margin))
        & (entropy <= float(max_entropy))
    )

    confidence_weight = ((stable_conf - sample_threshold) / (1.0 - sample_threshold + 1e-6)).clamp(0.0, 1.0)
    margin_weight = ((margin - float(min_margin)) / (1.0 - float(min_margin) + 1e-6)).clamp(0.0, 1.0)
    entropy_weight = ((float(max_entropy) - entropy) / max(float(max_entropy), 1e-6)).clamp(0.0, 1.0)
    pseudo_weight = torch.sqrt(confidence_weight * margin_weight * entropy_weight) * selected.float()
    align_mask = selected & (pseudo_weight >= float(align_min_weight))

    return (
        targets.cpu(),
        selected.cpu(),
        pseudo_weight.cpu(),
        align_mask.cpu(),
        agreement.cpu(),
        stable_conf.cpu(),
        margin.cpu(),
        entropy.cpu(),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data1", type=str, default="rafdb")
    parser.add_argument("--data2", type=str, default="fer")
    parser.add_argument("--idx", type=int, default=3)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v2"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--source_epochs", type=int, default=30)
    parser.add_argument("--w1", type=float, default=4.0)
    parser.add_argument("--w2", type=float, default=0.3)
    parser.add_argument("--w3", type=float, default=0.1)
    parser.add_argument("--phi", type=float, default=1.4)
    parser.add_argument("--source_root", type=str, default="/workspace/ttt/code/test-upload-clean/datesets/raf-basic")
    parser.add_argument("--target_root", type=str, default="/workspace/ttt/code/data/fer2013")
    parser.add_argument("--model_dir", type=str, default="./models")

    # Conservative target-stage defaults based on the observed confirmation-bias failure.
    parser.add_argument("--target_lr", type=float, default=1e-4)
    parser.add_argument("--target_lr_gamma", type=float, default=0.97)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--target_w2", type=float, default=0.03)
    parser.add_argument("--affinity_warmup", type=int, default=8)
    parser.add_argument("--affinity_ramp", type=int, default=5)
    parser.add_argument("--pseudo_ramp", type=int, default=10)
    parser.add_argument("--pseudo_min_threshold", type=float, default=0.90)
    parser.add_argument("--pseudo_max_threshold", type=float, default=0.97)
    parser.add_argument("--threshold_quantile", type=float, default=0.70)
    parser.add_argument("--min_margin", type=float, default=0.20)
    parser.add_argument("--max_entropy", type=float, default=0.55)
    parser.add_argument("--balance_power", type=float, default=0.0)
    parser.add_argument("--align_min_weight", type=float, default=0.50)
    parser.add_argument("--min_align_samples", type=int, default=8)

    args = parser.parse_args()
    if not 0.0 <= args.pseudo_min_threshold <= args.pseudo_max_threshold <= 0.99:
        parser.error("Require 0 <= pseudo_min_threshold <= pseudo_max_threshold <= 0.99.")
    if not 0.0 <= args.threshold_quantile <= 0.95:
        parser.error("threshold_quantile must be in [0, 0.95].")
    if not 0.0 <= args.min_margin < 1.0:
        parser.error("min_margin must be in [0, 1).")
    if not 0.0 < args.max_entropy <= 1.0:
        parser.error("max_entropy must be in (0, 1].")
    if args.source_epochs == 0 and not args.checkpoint:
        parser.error("source_epochs=0 requires --checkpoint.")
    return args


def make_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return {
        "train": transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(20), transforms.RandomCrop(224, padding=32)], p=0.5),
            transforms.ToTensor(), normalize, transforms.RandomErasing(scale=(0.02, 0.25)),
        ]),
        "weak": transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize,
        ]),
        "test": transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(), normalize,
        ]),
        "augment": transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(20), transforms.RandomCrop(224, padding=32)], p=0.5),
            RandAugmentMC(n=2, m=10), transforms.ToTensor(), normalize,
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ]),
    }


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


def scheduled_affinity_weight(epoch, maximum, warmup, ramp):
    if maximum <= 0 or epoch < warmup:
        return 0.0
    progress = float(epoch - warmup + 1) / float(max(ramp, 1))
    return float(maximum) * min(1.0, progress)


@torch.no_grad()
def evaluate(model, loader, num):
    model.eval()
    bingo_cnt = 0
    preds, labels = [], []
    for imgs, targets in loader:
        imgs = imgs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        out, _ = model(imgs, targets, None, mode="test")
        predicts = out.argmax(dim=1)
        bingo_cnt += predicts.eq(targets).sum().item()
        preds.append(predicts.cpu())
        labels.append(targets.cpu())
    util.make_confucion_matrix(preds, labels)
    return float(np.around(float(bingo_cnt) / float(max(num, 1)), 4))


def save_best(model, optimizer, path, epoch, accuracy, kind):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "accuracy": accuracy,
            "model_kind": kind,
        },
        path,
    )


def run_training():
    args = parse_args()
    model_path = os.path.join(args.model_dir, args.data1 + "_" + args.data2)
    os.makedirs(model_path, exist_ok=True)

    if args.backbone == "resnet50":
        train_batch, test_batch = 128, 100
    else:
        train_batch, test_batch = 128, 128

    tx = make_transforms()
    source_train = RafDataSet(args.source_root, "train", transform=tx["train"], strong_transform=None, basic_aug=False)
    target_train = FER(args.target_root, "train", transform=tx["weak"], strong_transform=tx["augment"], basic_aug=False)
    target_test = FER(args.target_root, "test", transform=tx["test"], strong_transform=None)

    if len(target_train) != 28709:
        print("WARNING: FER2013 train contains %d images; the paper uses 28709." % len(target_train))
    if len(target_test) != 3589:
        print("WARNING: FER2013 test contains %d images; the paper reports 3589." % len(target_test))

    train_loader_source = make_loader(source_train, train_batch, args.workers, True, True, 1)
    train_loader_target = make_loader(target_train, train_batch, args.workers, True, True, 2)
    threshold_loader_target = make_loader(target_train, train_batch, args.workers, False, False, 3)
    val_loader_target = make_loader(target_test, test_batch, args.workers, False, False, 4)

    model = Networks.Model(backbone=args.backbone, num_classes=7).cuda()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        model.load_state_dict(checkpoint["model"], strict=True)
        print("Loaded checkpoint:", args.checkpoint)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for i in range(args.source_epochs):
        model.train()
        correct = 0
        total = 0
        for imgs, targets in train_loader_source:
            imgs, targets = imgs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            output = model(imgs, targets, None, "train", "source", compute_affinity=args.w2 > 0)
            cls_loss = criterion(output[0], targets).mean()
            loss = cls_loss * args.w1 + output[1] * args.w2 + classifier_weight_loss(model) * args.w3
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite source loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += output[0].argmax(1).eq(targets).sum().item()
            total += targets.numel()
        scheduler.step()
        print("[Source %d] train_acc %.4f LR %.6f" % (i, correct / max(total, 1), optimizer.param_groups[0]["lr"]))

    source_fixed_path = os.path.join(model_path, "%s_%s_%s_source_final.pth" % (args.backbone, args.data1, args.data2))
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, source_fixed_path)
    print("Source checkpoint saved:", source_fixed_path)

    optimizer = torch.optim.Adam(model.parameters(), args.target_lr, weight_decay=1e-4)
    target_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.target_lr_gamma)
    teacher = copy.deepcopy(model).cuda().eval()
    for p in teacher.parameters():
        p.requires_grad = False

    baseline_student = evaluate(model, val_loader_target, len(target_test))
    baseline_teacher = evaluate(teacher, val_loader_target, len(target_test))
    print("[Epoch -1] Student accuracy: %.4f | EMA accuracy: %.4f" % (baseline_student, baseline_teacher))

    best_overall = max(baseline_student, baseline_teacher)
    best_student = baseline_student
    best_teacher = baseline_teacher
    best_path = os.path.join(model_path, "%s_%s_%s_best.pth" % (args.backbone, args.data1, args.data2))
    save_best(model if baseline_student >= baseline_teacher else teacher, optimizer, best_path, -1, best_overall,
              "student" if baseline_student >= baseline_teacher else "ema")

    source_train_iter = iter(train_loader_source)

    for i in range(args.epochs):
        thresholds, target_prior, support, agreement_full, predicted_count = estimate_target_statistics(
            teacher,
            threshold_loader_target,
            7,
            i,
            args.epochs,
            args.pseudo_min_threshold,
            args.pseudo_max_threshold,
            args.phi,
            args.threshold_quantile,
        )
        scheduled_w2 = scheduled_affinity_weight(i, args.target_w2, args.affinity_warmup, args.affinity_ramp)
        pseudo_scale = min(1.0, float(i + 1) / float(max(args.pseudo_ramp, 1)))

        print(
            "[Epoch %d] CATM %s support %s predicted %s agreement %.4f w2 %.4f pseudo_scale %.3f"
            % (
                i,
                [round(float(x), 4) for x in thresholds.tolist()],
                support.tolist(),
                predicted_count.tolist(),
                agreement_full,
                scheduled_w2,
                pseudo_scale,
            )
        )

        confident_num = 0
        align_num = 0
        pseudo_correct = 0
        pseudo_total = 0
        pseudo_class_total = np.zeros(7, dtype=np.int64)
        pseudo_class_correct = np.zeros(7, dtype=np.int64)
        pseudo_weight_sum = 0.0
        margin_sum = 0.0
        entropy_sum = 0.0
        applied_w2_sum = 0.0
        count = 0

        for imgs_w1, imgs_w2, imgs_aug, gt_target in train_loader_target:
            try:
                source_imgs, source_targets = next(source_train_iter)
            except StopIteration:
                source_train_iter = iter(train_loader_source)
                source_imgs, source_targets = next(source_train_iter)

            with torch.no_grad():
                out_w1, _ = teacher(imgs_w1.cuda(non_blocking=True), None, None, "test", "target")
                out_w2, _ = teacher(imgs_w2.cuda(non_blocking=True), None, None, "test", "target")
                (
                    targets,
                    selected_mask,
                    pseudo_weight,
                    align_mask,
                    agreement_mask,
                    stable_conf,
                    margin,
                    entropy,
                ) = build_pseudo_labels(
                    out_w1,
                    out_w2,
                    thresholds,
                    target_prior,
                    args.balance_power,
                    args.min_margin,
                    args.max_entropy,
                    args.align_min_weight,
                )

            selected_count = int(selected_mask.sum().item())
            align_count = int(align_mask.sum().item())
            confident_num += selected_count
            align_num += align_count

            if selected_count:
                selected_weights = pseudo_weight[selected_mask]
                pseudo_weight_sum += float(selected_weights.sum().item())
                margin_sum += float(margin[selected_mask].sum().item())
                entropy_sum += float(entropy[selected_mask].sum().item())

            # Target labels are used only for diagnostics, never for selection or loss.
            gt_cpu = gt_target.cpu().long()
            if selected_mask.any():
                pseudo_correct += targets[selected_mask].eq(gt_cpu[selected_mask]).sum().item()
                pseudo_total += selected_count
                for c in range(7):
                    cm = selected_mask & targets.eq(c)
                    n = int(cm.sum().item())
                    if n:
                        pseudo_class_total[c] += n
                        pseudo_class_correct[c] += targets[cm].eq(gt_cpu[cm]).sum().item()

            source_imgs = source_imgs.cuda(non_blocking=True)
            source_targets_cuda = source_targets.long().cuda(non_blocking=True)
            target_imgs = imgs_aug.cuda(non_blocking=True)
            target_targets_cuda = targets.long().cuda(non_blocking=True)
            pseudo_weight_cuda = pseudo_weight.cuda(non_blocking=True)
            align_mask_cuda = align_mask.cuda(non_blocking=True)
            n_source = source_imgs.shape[0]

            train_imgs = torch.cat((source_imgs, target_imgs), 0)
            train_targets = torch.cat((source_targets_cuda, target_targets_cuda), 0)
            train_con_idx = torch.cat((torch.ones(n_source, dtype=torch.bool, device="cuda"), align_mask_cuda), 0)

            mean_selected_weight = float(pseudo_weight[selected_mask].mean().item()) if selected_count else 0.0
            applied_w2 = scheduled_w2 if (
                align_count >= args.min_align_samples and mean_selected_weight >= args.align_min_weight
            ) else 0.0

            output = model(
                train_imgs,
                train_targets,
                train_con_idx,
                "train",
                "target",
                source_count=n_source,
                compute_affinity=applied_w2 > 0,
            )

            source_ce = criterion(output[0][:n_source], source_targets_cuda).mean()
            if selected_count > 0:
                target_losses = criterion(output[0][n_source:], target_targets_cuda)
                target_ce = (target_losses * pseudo_weight_cuda).sum() / pseudo_weight_cuda.sum().clamp_min(1e-6)
                cls_loss = source_ce + pseudo_scale * target_ce
            else:
                target_ce = output[0].new_zeros(())
                cls_loss = source_ce

            loss = args.w1 * cls_loss + output[1] * applied_w2 + classifier_weight_loss(model) * args.w3
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite target loss")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(model, teacher, args.ema_decay)

            applied_w2_sum += applied_w2
            count += 1

        target_scheduler.step()

        pseudo_acc = pseudo_correct / max(pseudo_total, 1)
        class_acc = [
            round(pseudo_class_correct[c] / float(pseudo_class_total[c]), 4) if pseudo_class_total[c] else 0.0
            for c in range(7)
        ]
        mean_weight = pseudo_weight_sum / max(pseudo_total, 1)
        mean_margin = margin_sum / max(pseudo_total, 1)
        mean_entropy = entropy_sum / max(pseudo_total, 1)

        print(
            "[Epoch %d] confident %d align %d pseudo_acc %.4f class_acc %s class_num %s "
            "mean_weight %.4f margin %.4f entropy %.4f applied_w2 %.4f"
            % (
                i,
                confident_num,
                align_num,
                pseudo_acc,
                class_acc,
                pseudo_class_total.tolist(),
                mean_weight,
                mean_margin,
                mean_entropy,
                applied_w2_sum / max(count, 1),
            )
        )

        student_acc = evaluate(model, val_loader_target, len(target_test))
        teacher_acc = evaluate(teacher, val_loader_target, len(target_test))
        print("[Epoch %d] Student accuracy: %.4f | EMA accuracy: %.4f" % (i, student_acc, teacher_acc))

        if student_acc > best_student:
            best_student = student_acc
            save_best(
                model,
                optimizer,
                os.path.join(model_path, "%s_%s_%s_student_best.pth" % (args.backbone, args.data1, args.data2)),
                i,
                student_acc,
                "student",
            )
        if teacher_acc > best_teacher:
            best_teacher = teacher_acc
            save_best(
                teacher,
                optimizer,
                os.path.join(model_path, "%s_%s_%s_ema_best.pth" % (args.backbone, args.data1, args.data2)),
                i,
                teacher_acc,
                "ema",
            )

        epoch_best = max(student_acc, teacher_acc)
        if epoch_best > best_overall:
            best_overall = epoch_best
            if teacher_acc >= student_acc:
                save_best(teacher, optimizer, best_path, i, teacher_acc, "ema")
            else:
                save_best(model, optimizer, best_path, i, student_acc, "student")
            print("Best checkpoint:", best_path, "acc", best_overall)

    print("Best student %.4f | Best EMA %.4f | best_acc %.4f" % (best_student, best_teacher, best_overall))


if __name__ == "__main__":
    run_training()
