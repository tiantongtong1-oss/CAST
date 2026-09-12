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
from dataset import RafDataSet, FER
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
    fc_weight = model.fc.weight
    fc_weight_norm = torch.norm(fc_weight, dim=1).unsqueeze(1).clamp_min(1e-12)
    fc_weight_ = fc_weight.mm(fc_weight.t())
    fc_weight_norm_ = fc_weight_norm.mm(fc_weight_norm.t()).clamp_min(1e-12)
    identity = torch.eye(fc_weight.shape[0], device=fc_weight.device)
    return torch.mean(((fc_weight_ / fc_weight_norm_ - identity) + 1.0) / 2.0)


@torch.no_grad()
def update_ema(student, teacher, decay):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)

    # Keep BatchNorm statistics smooth as well. Integer buffers such as
    # num_batches_tracked are copied exactly.
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if teacher_buffer.dtype.is_floating_point:
            teacher_buffer.data.mul_(decay).add_(student_buffer.data, alpha=1.0 - decay)
        else:
            teacher_buffer.data.copy_(student_buffer.data)


def adjust_probabilities(prob, prior, balance_power):
    """Mild distribution alignment to prevent one-class pseudo-label collapse."""
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
    threshold_quantile,
    balance_power,
    phi,
):
    """
    Estimate one class-wise threshold vector for the whole target epoch.

    The previous implementation created threshold_loader_target but still
    recomputed CATM thresholds inside every mini-batch. Here we use the full
    target set and two teacher views, so thresholds are much less noisy.
    """
    teacher.eval()
    cached_prob_1 = []
    cached_prob_2 = []
    prior_sum = torch.zeros(class_num, dtype=torch.float64)
    sample_count = 0

    for imgs_w1, imgs_w2, _, _ in loader:
        out_w1, _ = teacher(imgs_w1.cuda(non_blocking=True), None, None, "test", "target")
        out_w2, _ = teacher(imgs_w2.cuda(non_blocking=True), None, None, "test", "target")

        prob_1 = F.softmax(out_w1, dim=1).cpu()
        prob_2 = F.softmax(out_w2, dim=1).cpu()
        avg_prob = 0.5 * (prob_1 + prob_2)

        cached_prob_1.append(prob_1)
        cached_prob_2.append(prob_2)
        prior_sum += avg_prob.double().sum(dim=0)
        sample_count += avg_prob.shape[0]

    prior = (prior_sum / max(sample_count, 1)).float()
    prior = prior / prior.sum().clamp_min(1e-12)

    class_confidence = [[] for _ in range(class_num)]
    agreement_num = 0
    agreement_total = 0

    for prob_1, prob_2 in zip(cached_prob_1, cached_prob_2):
        prob_1 = adjust_probabilities(prob_1, prior, balance_power)
        prob_2 = adjust_probabilities(prob_2, prior, balance_power)
        avg_prob = 0.5 * (prob_1 + prob_2)

        pred_1 = prob_1.argmax(dim=1)
        pred_2 = prob_2.argmax(dim=1)
        targets = avg_prob.argmax(dim=1)

        target_conf_1 = prob_1.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_conf_2 = prob_2.gather(1, targets.unsqueeze(1)).squeeze(1)
        stable_conf = torch.minimum(target_conf_1, target_conf_2)
        agreement = pred_1.eq(pred_2)

        agreement_num += int(agreement.sum().item())
        agreement_total += int(agreement.numel())

        for c in range(class_num):
            values = stable_conf[(targets == c) & agreement]
            if values.numel() > 0:
                class_confidence[c].append(values)

    progress = float(epoch) / float(max(total_epochs - 1, 1))
    # Become moderately stricter over time. Unlike the old total/(total-i)
    # multiplier, this does not force nearly every class to the 0.9 cap.
    q = float(np.clip(threshold_quantile + 0.10 * progress, 0.0, 0.95))
    phi_scale = float(np.clip(phi / 1.4, 0.90, 1.10))

    thresholds = torch.full((class_num,), max_threshold, dtype=torch.float32)
    support = torch.zeros(class_num, dtype=torch.long)

    for c in range(class_num):
        if not class_confidence[c]:
            continue

        values = torch.cat(class_confidence[c], dim=0)
        support[c] = values.numel()

        if values.numel() < 8:
            threshold = max_threshold
        else:
            threshold = float(torch.quantile(values, q).item()) * phi_scale
            threshold = float(np.clip(threshold, min_threshold, max_threshold))

        thresholds[c] = threshold

    agreement_rate = agreement_num / max(agreement_total, 1)
    return thresholds, prior, support, agreement_rate


@torch.no_grad()
def build_pseudo_labels(
    out_w1,
    out_w2,
    thresholds,
    prior,
    balance_power,
    align_min_weight,
):
    prob_1 = adjust_probabilities(F.softmax(out_w1, dim=1), prior, balance_power)
    prob_2 = adjust_probabilities(F.softmax(out_w2, dim=1), prior, balance_power)
    avg_prob = 0.5 * (prob_1 + prob_2)

    pred_1 = prob_1.argmax(dim=1)
    pred_2 = prob_2.argmax(dim=1)
    targets = avg_prob.argmax(dim=1)

    target_conf_1 = prob_1.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_conf_2 = prob_2.gather(1, targets.unsqueeze(1)).squeeze(1)
    stable_conf = torch.minimum(target_conf_1, target_conf_2)

    thresholds = thresholds.to(out_w1.device)
    sample_threshold = thresholds[targets]
    agreement = pred_1.eq(pred_2)

    # A strict agreement path plus a very-high-confidence escape hatch.
    # The latter avoids completely starving a class when the two views differ
    # only marginally.
    ultra_threshold = torch.maximum(
        sample_threshold + 0.05,
        torch.full_like(sample_threshold, 0.97),
    )
    selected = (agreement & (stable_conf >= sample_threshold)) | (stable_conf >= ultra_threshold)

    pseudo_weight = (stable_conf - sample_threshold) / (1.0 - sample_threshold + 1e-6)
    pseudo_weight = pseudo_weight.clamp(0.0, 1.0)
    pseudo_weight = torch.sqrt(pseudo_weight) * selected.float()

    align_mask = selected & (pseudo_weight >= align_min_weight)

    return (
        targets.cpu(),
        selected.cpu(),
        pseudo_weight.cpu(),
        align_mask.cpu(),
        agreement.cpu(),
        stable_conf.cpu(),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data1", type=str, default="rafdb", help="source data")
    parser.add_argument("--data2", type=str, default="fer", help="target data")
    parser.add_argument("--idx", type=int, default=3, help="cross-validation index")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="load model")
    parser.add_argument(
        "--source_checkpoint",
        type=str,
        default=None,
        help="load a finished model and skip source pre-training",
    )
    parser.add_argument("--backbone", type=str, default="resnet18", help="resnet18, resnet50 or mobilenet_v2")
    parser.add_argument("--lr", type=float, default=0.001, help="source learning rate")
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--epochs", type=int, default=30, help="target adaptation epochs")
    parser.add_argument("--source_epochs", type=int, default=30)
    parser.add_argument("--w1", type=float, default=4.0, help="classification loss weight")
    parser.add_argument("--w2", type=float, default=0.3, help="source affinity loss weight")
    parser.add_argument("--w3", type=float, default=0.1, help="classifier weight loss weight")
    parser.add_argument("--phi", type=float, default=1.4, help="CATM threshold aggressiveness")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--target_w2", type=float, default=0.03)

    # Stability controls for target adaptation.
    parser.add_argument("--ema_floor", type=float, default=0.999)
    parser.add_argument("--teacher_warmup", type=int, default=5)
    parser.add_argument("--target_lr", type=float, default=2e-4)
    parser.add_argument("--target_lr_gamma", type=float, default=0.97)
    parser.add_argument("--target_w2_cap", type=float, default=0.01)
    parser.add_argument("--pseudo_min_threshold", type=float, default=0.80)
    parser.add_argument("--pseudo_max_threshold", type=float, default=0.95)
    parser.add_argument("--pseudo_quantile", type=float, default=0.40)
    parser.add_argument("--balance_power", type=float, default=0.20)
    parser.add_argument("--align_min_weight", type=float, default=0.70)
    parser.add_argument("--min_align_samples", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def make_transforms():
    return {
        "train": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32),
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ]),
        "test": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "augment": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32),
            ], p=0.5),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


def run_training():
    args = parse_args()

    if args.checkpoint and args.source_checkpoint:
        raise ValueError("Use only one of --checkpoint and --source_checkpoint")
    if args.pseudo_min_threshold >= args.pseudo_max_threshold:
        raise ValueError("pseudo_min_threshold must be smaller than pseudo_max_threshold")
    if args.source_epochs < 0 or args.epochs < 0:
        raise ValueError("source_epochs and epochs must be non-negative")

    model_path = os.path.join(args.output_dir, args.data1 + "_" + args.data2)
    if args.run_name:
        if os.path.basename(args.run_name) != args.run_name:
            raise ValueError("run_name must be a directory name")
        model_path = os.path.join(model_path, args.run_name)
        os.makedirs(model_path, exist_ok=False)
    else:
        os.makedirs(model_path, exist_ok=True)
    print("Run directory:", model_path)

    print("---------------------------------------------------------------------------------------")
    print("Training %s with source data %s and target data %s, idx %s:" % (
        args.backbone, args.data1, args.data2, args.idx
    ))
    print("w1:%s        w2:%s      w3:%s" % (args.w1, args.w2, args.w3))
    print("---------------------------------------------------------------------------------------")

    if args.backbone == "resnet18":
        train_batch, test_batch, pre_epochs = 128, 128, 30
    elif args.backbone == "resnet50":
        train_batch, test_batch, pre_epochs = 128, 100, 30
    elif args.backbone == "mobilenet_v2":
        train_batch, test_batch, pre_epochs = 128, 128, 30
    else:
        raise ValueError("Backbone Error!")

    data_transforms = make_transforms()

    if args.data1 != "rafdb":
        raise ValueError("Please input right source data")

    # The source strong view was never used by training. Avoid generating a
    # second augmentation from an already-normalized tensor in RafDataSet.
    source_train = RafDataSet(
        "/workspace/ttt/code/test-upload-clean/datesets/raf-basic",
        phase="train",
        transform=data_transforms["train"],
        strong_transform=None,
        basic_aug=False,
    )
    source_test = RafDataSet(
        "/workspace/ttt/code/test-upload-clean/datesets/raf-basic",
        phase="test",
        transform=data_transforms["test"],
        strong_transform=None,
    )

    if args.data2 != "fer":
        raise ValueError("Please input right target data")

    target_train = FER(
        "/workspace/ttt/code/data/fer2013",
        phase="train",
        transform=data_transforms["train"],
        strong_transform=data_transforms["augment"],
        basic_aug=False,
    )
    target_test = FER(
        "/workspace/ttt/code/data/fer2013",
        phase="test",
        transform=data_transforms["test"],
        strong_transform=None,
    )

    class_num = 7
    target_val_num = len(target_test)

    train_loader_source = make_loader(source_train, train_batch, args.workers, True, True, 1)
    train_loader_target = make_loader(target_train, train_batch, args.workers, True, True, 2)
    threshold_loader_target = make_loader(target_train, train_batch, args.workers, False, False, 3)
    val_loader_target = make_loader(target_test, test_batch, args.workers, False, False, 4)

    model = Networks.Model(backbone=args.backbone, num_classes=class_num)

    initial_checkpoint = args.source_checkpoint or args.checkpoint
    if initial_checkpoint:
        print("Loading checkpoint:", initial_checkpoint)
        checkpoint = torch.load(initial_checkpoint)
        model.load_state_dict(checkpoint["model"], strict=True)

    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # ------------------------------------------------------------------
    # Source pre-training
    # ------------------------------------------------------------------
    best_acc = 0.0
    for i in range(args.source_epochs if not args.source_checkpoint else 0):
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        count = 0
        bingo_cnt = 0
        model.train()

        for imgs, targets in train_loader_source:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            output = model(imgs, targets, None, "train", "source")
            cls_loss = criterion(output[0], targets).mean()
            aff_loss = output[1]
            weight_loss = classifier_weight_loss(model)
            loss = cls_loss * args.w1 + aff_loss * args.w2 + weight_loss * args.w3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicts = output[0].argmax(dim=1)
            bingo_cnt += predicts.eq(targets).sum().item()
            train_loss1 += cls_loss.item()
            train_loss2 += aff_loss.item()
            train_loss3 += weight_loss.item()
            count += 1

        train_acc = bingo_cnt / float(max(count * train_batch, 1))
        print(
            "[Epoch %d] Training accuracy: %.4f.   Classification Loss: %.3f   "
            "Affinity Loss: %.3f  Weight Loss: %.3f  LR: %.6f"
            % (
                i,
                train_acc,
                train_loss1 / max(count, 1),
                train_loss2 / max(count, 1),
                train_loss3 / max(count, 1),
                optimizer.param_groups[0]["lr"],
            )
        )

        scheduler.step()
        best_acc = test(
            model,
            optimizer,
            val_loader_target,
            criterion,
            target_val_num,
            best_acc,
            model_path,
            i,
            args,
        )

    # Reload the best source checkpoint, or use the supplied finished model
    # directly. The latter avoids another 30 source epochs changing the start.
    if args.source_checkpoint:
        source_best_acc = test(
            model,
            optimizer,
            val_loader_target,
            criterion,
            target_val_num,
            0.0,
            model_path,
            -1,
            args,
        )
        print("Source initialization target accuracy:", source_best_acc)
    else:
        source_best_acc = best_acc
        source_checkpoint_path = os.path.join(
            model_path,
            "%s_%s_%s_%s.pth" % (args.backbone, args.data1, args.data2, source_best_acc),
        )
        print("Loading best source checkpoint:", source_checkpoint_path)
        checkpoint = torch.load(source_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Best source checkpoint loaded, acc:", source_best_acc)

    source_fixed_path = os.path.join(
        model_path,
        "%s_%s_%s_source_best.pth" % (args.backbone, args.data1, args.data2),
    )
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, source_fixed_path)
    print("Source checkpoint saved:", source_fixed_path)

    teacher = copy.deepcopy(model).cuda()
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Remove stale source-phase LR coupling. Keep Adam moments, but use a
    # controlled target LR and a separate target scheduler.
    for group in optimizer.param_groups:
        group["lr"] = min(float(group["lr"]), args.target_lr)
        group["initial_lr"] = group["lr"]
    target_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=args.target_lr_gamma,
    )

    effective_ema_decay = max(args.ema_decay, args.ema_floor)
    requested_target_w2 = max(args.target_w2, 0.0)
    target_w2_ceiling = min(requested_target_w2, args.target_w2_cap)

    print(
        "Target adaptation controls: LR=%.6f EMA=%.5f (requested %.5f) target_w2<=%.4f (requested %.4f)"
        % (
            optimizer.param_groups[0]["lr"],
            effective_ema_decay,
            args.ema_decay,
            target_w2_ceiling,
            args.target_w2,
        )
    )

    # Preserve the initialization as the best candidate when adaptation regresses.
    best_acc = source_best_acc
    source_train_iter = iter(train_loader_source)

    # ------------------------------------------------------------------
    # Stable target adaptation
    # ------------------------------------------------------------------
    for i in range(args.epochs):
        thresholds, target_prior, threshold_support, threshold_agreement = estimate_target_statistics(
            teacher=teacher,
            loader=threshold_loader_target,
            class_num=class_num,
            epoch=i,
            total_epochs=args.epochs,
            min_threshold=args.pseudo_min_threshold,
            max_threshold=args.pseudo_max_threshold,
            threshold_quantile=args.pseudo_quantile,
            balance_power=args.balance_power,
            phi=args.phi,
        )

        if i < 8:
            scheduled_w2 = 0.0
        else:
            ramp = min(1.0, float(i - 7) / 7.0)
            scheduled_w2 = target_w2_ceiling * ramp

        print("[Epoch %d] Target prior: %s" % (
            i, [round(float(x), 4) for x in target_prior.tolist()]
        ))
        print("[Epoch %d] CATM thresholds: %s | support: %s | full-set agreement: %.4f" % (
            i,
            [round(float(x), 4) for x in thresholds.tolist()],
            threshold_support.tolist(),
            threshold_agreement,
        ))
        print("[Epoch %d] Scheduled target w2: %.4f" % (i, scheduled_w2))

        train_loss1 = 0.0
        train_loss2 = 0.0
        count = 0
        confident_num = 0
        align_num = 0
        agreement_num = 0
        agreement_total = 0
        pseudo_weight_sum = 0.0
        pseudo_weight_num = 0
        pseudo_correct = 0
        pseudo_total = 0
        pseudo_class_correct = np.zeros(class_num, dtype=np.int64)
        pseudo_class_total = np.zeros(class_num, dtype=np.int64)
        applied_w2_sum = 0.0

        for imgs_w1, imgs_w2, imgs_aug, gt_target in train_loader_target:
            try:
                source_imgs, source_targets = next(source_train_iter)
            except StopIteration:
                source_train_iter = iter(train_loader_source)
                source_imgs, source_targets = next(source_train_iter)

            teacher.eval()
            with torch.no_grad():
                out_w1, _ = teacher(
                    imgs_w1.cuda(non_blocking=True), None, None, "test", "target"
                )
                out_w2, _ = teacher(
                    imgs_w2.cuda(non_blocking=True), None, None, "test", "target"
                )

                (
                    targets,
                    selected_mask,
                    pseudo_weight,
                    align_mask,
                    agreement_mask,
                    stable_conf,
                ) = build_pseudo_labels(
                    out_w1,
                    out_w2,
                    thresholds,
                    target_prior,
                    args.balance_power,
                    args.align_min_weight,
                )

            agreement_num += int(agreement_mask.sum().item())
            agreement_total += int(agreement_mask.numel())
            confident_num += int(selected_mask.sum().item())
            align_count = int(align_mask.sum().item())
            align_num += align_count

            selected_weight = pseudo_weight[selected_mask]
            if selected_weight.numel() > 0:
                pseudo_weight_sum += selected_weight.sum().item()
                pseudo_weight_num += selected_weight.numel()

            # Diagnostic only. Target labels never enter thresholding or loss.
            gt_cpu = gt_target.cpu().long()
            if selected_mask.any():
                pseudo_correct += targets[selected_mask].eq(gt_cpu[selected_mask]).sum().item()
                pseudo_total += int(selected_mask.sum().item())
                for c in range(class_num):
                    class_mask = selected_mask & targets.eq(c)
                    n_c = int(class_mask.sum().item())
                    if n_c > 0:
                        pseudo_class_total[c] += n_c
                        pseudo_class_correct[c] += targets[class_mask].eq(gt_cpu[class_mask]).sum().item()

            model.train()
            source_imgs = source_imgs.cuda(non_blocking=True)
            source_targets = source_targets.long()
            target_imgs = imgs_aug.cuda(non_blocking=True)

            source_con_idx = torch.ones(source_imgs.shape[0], dtype=torch.float32)
            train_imgs = torch.cat((source_imgs, target_imgs), dim=0)
            train_targets = torch.cat((source_targets, targets), dim=0).cuda(non_blocking=True)
            train_con_idx = torch.cat((source_con_idx, align_mask.float()), dim=0).cuda(non_blocking=True)
            train_cls_weight = torch.cat(
                (torch.ones(source_imgs.shape[0]), pseudo_weight.float()),
                dim=0,
            ).cuda(non_blocking=True)

            output = model(train_imgs, train_targets, train_con_idx, "train", "target")
            per_sample_loss = criterion(output[0], train_targets)

            # Do not dilute source replay by zero-weight target samples.
            cls_loss = (per_sample_loss * train_cls_weight).sum() / train_cls_weight.sum().clamp_min(1.0)
            aff_loss = output[1]
            weight_loss = classifier_weight_loss(model)

            # DDRL affinity is useful only when enough high-reliability target
            # samples are present. This prevents a few noisy pseudo labels from
            # producing a large negative affinity term.
            applied_w2 = scheduled_w2 if align_count >= args.min_align_samples else 0.0
            loss = cls_loss * args.w1 + aff_loss * applied_w2 + weight_loss * args.w3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i >= args.teacher_warmup:
                update_ema(model, teacher, effective_ema_decay)

            train_loss1 += cls_loss.item()
            train_loss2 += aff_loss.item()
            applied_w2_sum += applied_w2
            count += 1

        target_scheduler.step()

        mean_pseudo_weight = pseudo_weight_sum / max(pseudo_weight_num, 1)
        agreement_rate = agreement_num / max(agreement_total, 1)
        pseudo_acc = pseudo_correct / max(pseudo_total, 1)
        pseudo_class_acc = []
        for c in range(class_num):
            if pseudo_class_total[c] > 0:
                pseudo_class_acc.append(round(
                    pseudo_class_correct[c] / float(pseudo_class_total[c]), 4
                ))
            else:
                pseudo_class_acc.append(0.0)

        print("[Epoch %d] Mean pseudo weight: %.4f" % (i, mean_pseudo_weight))
        print("[Epoch %d] Two-view Agreement: %.4f (%d/%d)" % (
            i, agreement_rate, agreement_num, agreement_total
        ))
        print("[Epoch %d] Pseudo Acc: %.4f | Pseudo class acc: %s | Pseudo class num: %s" % (
            i, pseudo_acc, pseudo_class_acc, pseudo_class_total.tolist()
        ))
        print(
            "[Epoch %d] Confident_Num: %d Align_Num: %d Classification Loss: %.3f "
            "Affinity Loss: %.3f Applied_w2: %.4f LR: %.6f"
            % (
                i,
                confident_num,
                align_num,
                train_loss1 / max(count, 1),
                train_loss2 / max(count, 1),
                applied_w2_sum / max(count, 1),
                optimizer.param_groups[0]["lr"],
            )
        )

        best_acc = test(
            model,
            optimizer,
            val_loader_target,
            criterion,
            target_val_num,
            best_acc,
            model_path,
            i,
            args,
        )

    print("best_acc %s " % str(best_acc))


def test(model, optimizer, val_loader_target, criterion, num, best_acc, path, epoch, args):
    with torch.no_grad():
        val_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        preds = []
        labels = []
        model.eval()

        for imgs, targets in val_loader_target:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            out, _ = model(imgs, targets, None, mode="test")
            loss = criterion(out, targets).mean()

            val_loss += loss.item()
            iter_cnt += 1
            predicts = out.argmax(dim=1)
            bingo_cnt += predicts.eq(targets).sum().item()
            preds.append(predicts.cpu())
            labels.append(targets.cpu())

        val_loss /= max(iter_cnt, 1)
        val_acc = np.around(float(bingo_cnt) / float(num), 4)
        print("[Epoch %d] Target Validation accuracy:%.4f.  Loss:%.3f" % (
            epoch, val_acc, val_loss
        ))

        # Keep the original confusion-matrix side effect for compatibility.
        util.make_confucion_matrix(preds, labels)
        mean_acc = val_acc

        if mean_acc > best_acc:
            old_path = os.path.join(
                path,
                "%s_%s_%s_%s.pth" % (args.backbone, args.data1, args.data2, best_acc),
            )
            try:
                os.remove(old_path)
            except OSError:
                pass

            best_acc = mean_acc
            save_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_path = os.path.join(
                path,
                "%s_%s_%s_%s.pth" % (args.backbone, args.data1, args.data2, best_acc),
            )
            torch.save(save_data, save_path)
            print("best_acc %s " % str(best_acc))

    return best_acc


if __name__ == "__main__":
    run_training()
