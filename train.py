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
    fc_weight = model.fc.weight
    normalized = F.normalize(fc_weight, dim=1)
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
def estimate_target_statistics(teacher, loader, class_num, epoch, total_epochs, min_threshold, max_threshold, phi):
    teacher.eval()
    confidence_sum = torch.zeros(class_num, dtype=torch.float64)
    class_count = torch.zeros(class_num, dtype=torch.long)
    prior_sum = torch.zeros(class_num, dtype=torch.float64)
    sample_count = 0
    agreement_num = 0
    agreement_total = 0

    for imgs_w1, imgs_w2, _, _ in loader:
        out_w1, _ = teacher(imgs_w1.cuda(non_blocking=True), None, None, "test", "target")
        out_w2, _ = teacher(imgs_w2.cuda(non_blocking=True), None, None, "test", "target")
        prob_1 = F.softmax(out_w1, dim=1).cpu()
        prob_2 = F.softmax(out_w2, dim=1).cpu()
        avg_prob = 0.5 * (prob_1 + prob_2)
        confidence, targets = avg_prob.max(dim=1)
        confidence_sum.scatter_add_(0, targets, confidence.double())
        class_count += torch.bincount(targets, minlength=class_num)
        prior_sum += avg_prob.double().sum(dim=0)
        sample_count += avg_prob.shape[0]
        agreement = prob_1.argmax(dim=1).eq(prob_2.argmax(dim=1))
        agreement_num += int(agreement.sum().item())
        agreement_total += int(agreement.numel())

    mean_confidence = confidence_sum / class_count.clamp_min(1).double()
    stage_factor = float(total_epochs) / float(max(total_epochs - epoch, 1))
    thresholds = mean_confidence.float() * float(phi) * stage_factor
    thresholds = thresholds.clamp(min=float(min_threshold), max=float(max_threshold))
    thresholds[class_count == 0] = float(max_threshold)
    prior = (prior_sum / max(sample_count, 1)).float()
    prior = prior / prior.sum().clamp_min(1e-12)
    agreement_rate = agreement_num / max(agreement_total, 1)
    return thresholds, prior, class_count, agreement_rate


@torch.no_grad()
def build_pseudo_labels(out_w1, out_w2, thresholds, prior, balance_power):
    prob_1 = adjust_probabilities(F.softmax(out_w1, dim=1), prior, balance_power)
    prob_2 = adjust_probabilities(F.softmax(out_w2, dim=1), prior, balance_power)
    avg_prob = 0.5 * (prob_1 + prob_2)
    pred_1 = prob_1.argmax(dim=1)
    pred_2 = prob_2.argmax(dim=1)
    targets = avg_prob.argmax(dim=1)
    target_conf_1 = prob_1.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_conf_2 = prob_2.gather(1, targets.unsqueeze(1)).squeeze(1)
    stable_conf = torch.minimum(target_conf_1, target_conf_2)
    sample_threshold = thresholds.to(out_w1.device)[targets]
    agreement = pred_1.eq(pred_2)
    selected = agreement & (stable_conf >= sample_threshold)
    return targets.cpu(), selected.cpu(), agreement.cpu(), stable_conf.cpu()


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
    parser.add_argument("--target_lr", type=float, default=3e-4)
    parser.add_argument("--target_lr_gamma", type=float, default=0.97)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--target_w2", type=float, default=0.3)
    parser.add_argument("--affinity_warmup", type=int, default=3)
    parser.add_argument("--affinity_ramp", type=int, default=5)
    parser.add_argument("--pseudo_ramp", type=int, default=5)
    parser.add_argument("--pseudo_min_threshold", type=float, default=0.0)
    parser.add_argument("--pseudo_max_threshold", type=float, default=0.9)
    parser.add_argument("--balance_power", type=float, default=0.0)
    parser.add_argument("--min_align_samples", type=int, default=8)
    args = parser.parse_args()
    if not 0.0 <= args.pseudo_min_threshold <= args.pseudo_max_threshold <= 0.99:
        parser.error("Require 0 <= pseudo_min_threshold <= pseudo_max_threshold <= 0.99.")
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
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
                                       pin_memory=True, drop_last=drop_last, worker_init_fn=seed_worker,
                                       generator=generator)


def scheduled_affinity_weight(epoch, maximum, warmup, ramp):
    if maximum <= 0 or epoch < warmup:
        return 0.0
    progress = float(epoch - warmup + 1) / float(max(ramp, 1))
    return float(maximum) * min(1.0, progress)


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
        print("WARNING: FER2013 test contains %d images; paper reports 3589." % len(target_test))

    train_loader_source = make_loader(source_train, train_batch, args.workers, True, True, 1)
    train_loader_target = make_loader(target_train, train_batch, args.workers, True, True, 2)
    threshold_loader_target = make_loader(target_train, train_batch, args.workers, False, False, 3)
    val_loader_target = make_loader(target_test, test_batch, args.workers, False, False, 4)

    model = Networks.Model(backbone=args.backbone, num_classes=7).cuda()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        model.load_state_dict(checkpoint["model"], strict=True)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for i in range(args.source_epochs):
        model.train()
        for imgs, targets in train_loader_source:
            imgs, targets = imgs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            output = model(imgs, targets, None, "train", "source", compute_affinity=args.w2 > 0)
            cls_loss = criterion(output[0], targets).mean()
            loss = cls_loss * args.w1 + output[1] * args.w2 + classifier_weight_loss(model) * args.w3
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite source loss")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        print("[Source %d] LR %.6f" % (i, optimizer.param_groups[0]["lr"]))

    source_fixed_path = os.path.join(model_path, "%s_%s_%s_source_final.pth" % (args.backbone, args.data1, args.data2))
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, source_fixed_path)
    print("Source checkpoint saved:", source_fixed_path)

    optimizer = torch.optim.Adam(model.parameters(), args.target_lr, weight_decay=1e-4)
    target_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.target_lr_gamma)
    teacher = copy.deepcopy(model).cuda().eval()
    for p in teacher.parameters(): p.requires_grad = False

    best_acc = test(model, optimizer, val_loader_target, criterion, len(target_test), 0.0, model_path, -1, args)
    source_train_iter = iter(train_loader_source)

    for i in range(args.epochs):
        thresholds, target_prior, support, agreement_full = estimate_target_statistics(
            teacher, threshold_loader_target, 7, i, args.epochs,
            args.pseudo_min_threshold, args.pseudo_max_threshold, args.phi)
        scheduled_w2 = scheduled_affinity_weight(i, args.target_w2, args.affinity_warmup, args.affinity_ramp)
        pseudo_scale = min(1.0, float(i + 1) / float(max(args.pseudo_ramp, 1)))
        print("[Epoch %d] CATM %s support %s agreement %.4f w2 %.4f" %
              (i, [round(float(x), 4) for x in thresholds.tolist()], support.tolist(), agreement_full, scheduled_w2))

        confident_num = 0
        pseudo_correct = 0
        pseudo_total = 0
        pseudo_class_total = np.zeros(7, dtype=np.int64)
        pseudo_class_correct = np.zeros(7, dtype=np.int64)
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
                targets, selected_mask, _, _ = build_pseudo_labels(out_w1, out_w2, thresholds, target_prior, args.balance_power)

            selected_count = int(selected_mask.sum().item())
            confident_num += selected_count
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
            selected_cuda = selected_mask.cuda(non_blocking=True)
            n_source = source_imgs.shape[0]
            train_imgs = torch.cat((source_imgs, target_imgs), 0)
            train_targets = torch.cat((source_targets_cuda, target_targets_cuda), 0)
            train_con_idx = torch.cat((torch.ones(n_source, dtype=torch.bool, device="cuda"), selected_cuda), 0)

            applied_w2 = scheduled_w2 if selected_count >= args.min_align_samples else 0.0
            output = model(train_imgs, train_targets, train_con_idx, "train", "target",
                           source_count=n_source, compute_affinity=applied_w2 > 0)
            source_ce = criterion(output[0][:n_source], source_targets_cuda).mean()
            if selected_count > 0:
                tl = criterion(output[0][n_source:], target_targets_cuda)
                target_ce = (tl * selected_cuda.float()).sum() / selected_cuda.float().sum().clamp_min(1.0)
                cls_loss = 0.5 * (source_ce + pseudo_scale * target_ce)
            else:
                cls_loss = source_ce

            loss = cls_loss * args.w1 + output[1] * applied_w2 + classifier_weight_loss(model) * args.w3
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite target loss")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            update_ema(model, teacher, args.ema_decay)
            applied_w2_sum += applied_w2
            count += 1

        target_scheduler.step()
        pseudo_acc = pseudo_correct / max(pseudo_total, 1)
        class_acc = [round(pseudo_class_correct[c] / float(pseudo_class_total[c]), 4) if pseudo_class_total[c] else 0.0 for c in range(7)]
        print("[Epoch %d] confident %d pseudo_acc %.4f class_acc %s class_num %s applied_w2 %.4f" %
              (i, confident_num, pseudo_acc, class_acc, pseudo_class_total.tolist(), applied_w2_sum / max(count, 1)))
        best_acc = test(model, optimizer, val_loader_target, criterion, len(target_test), best_acc, model_path, i, args)

    print("best_acc %s" % best_acc)


def test(model, optimizer, val_loader_target, criterion, num, best_acc, path, epoch, args):
    with torch.no_grad():
        bingo_cnt = 0
        preds, labels = [], []
        model.eval()
        for imgs, targets in val_loader_target:
            imgs, targets = imgs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            out, _ = model(imgs, targets, None, mode="test")
            predicts = out.argmax(dim=1)
            bingo_cnt += predicts.eq(targets).sum().item()
            preds.append(predicts.cpu()); labels.append(targets.cpu())
        val_acc = np.around(float(bingo_cnt) / float(max(num, 1)), 4)
        print("[Epoch %d] Target Validation accuracy:%.4f" % (epoch, val_acc))
        util.make_confucion_matrix(preds, labels)
        if float(val_acc) > best_acc:
            best_acc = float(val_acc)
            save_path = os.path.join(path, "%s_%s_%s_best.pth" % (args.backbone, args.data1, args.data2))
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "accuracy": best_acc}, save_path)
            print("Best checkpoint:", save_path)
    return best_acc


if __name__ == "__main__":
    run_training()
