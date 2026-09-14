import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import Networks
from dataset import RafDataSet, FER
from ema_utils import (
    PrototypeMemory,
    create_ema_teacher,
    align_dual_view_probabilities,
    select_dual_view_pseudo_labels,
    weighted_mean_loss,
    update_ema_teacher,
)
import image_utils as util
from randaugment import RandAugmentMC


seed = 1314
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(seed + worker_id)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data1', type=str, default='rafdb', help='source data')
    parser.add_argument('--data2', type=str, default='fer', help='target data')
    parser.add_argument('--source_path', type=str,
                        default='/workspace/ttt/code/test-upload-clean/datesets/raf-basic')
    parser.add_argument('--target_path', type=str,
                        default='/workspace/ttt/code/data/fer2013')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='load model')

    # This innovation branch is centered on MobileNetV2. Other backbones are
    # retained for controlled ablations, but MobileNetV2 is the default.
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        help='mobilenet_v2 (default), resnet18 or resnet50')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--target_lr', type=float, default=None,
                        help='fresh target-stage learning rate; defaults to --lr')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--no_pretrained', action='store_true',
                        help='disable ImageNet weight download for offline checks')
    parser.add_argument('--output_dir', default='./models')
    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument('--pre_epochs', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=30)

    # Original CAST loss weights kept for comparability with the baseline.
    parser.add_argument('--w1', type=float, default=4.0,
                        help='classification loss weight')
    parser.add_argument('--w2', type=float, default=0.3,
                        help='DDRL feature loss weight')
    parser.add_argument('--w3', type=float, default=0.1,
                        help='classifier modulation loss weight')
    parser.add_argument('--phi', type=float, default=1.4,
                        help='class-wise threshold scale')

    # Dual-view EMA teacher.
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--teacher_temperature', type=float, default=2.0,
                        help='soften EMA teacher confidence before pseudo labeling')
    parser.add_argument('--consistency_min_conf', type=float, default=0.50)
    parser.add_argument('--fallback_conf', type=float, default=0.90)

    # Global class-wise threshold and class-distribution correction.
    parser.add_argument('--threshold_min', type=float, default=0.55)
    parser.add_argument('--threshold_max', type=float, default=0.95)
    parser.add_argument('--threshold_stage_gain', type=float, default=0.05)
    parser.add_argument('--distribution_power', type=float, default=0.5)
    parser.add_argument('--distribution_ratio_max', type=float, default=3.0)

    # Improved source/target classification balancing.
    parser.add_argument('--lambda_target', type=float, default=1.0)
    parser.add_argument('--target_ramp_epochs', type=int, default=5)
    parser.add_argument('--pseudo_weight_max', type=float, default=2.0)

    # Prototype-based target affinity loss with a warm-up/ramp upper bound.
    parser.add_argument('--lambda_aff', type=float, default=0.10,
                        help='maximum prototype affinity weight')
    parser.add_argument('--aff_warmup_epochs', type=int, default=2)
    parser.add_argument('--aff_ramp_epochs', type=int, default=5)
    parser.add_argument('--prototype_momentum', type=float, default=0.9)
    parser.add_argument('--prototype_margin', type=float, default=0.2)

    # Stability controls.
    parser.add_argument('--class_weight_max', type=float, default=1.0)
    parser.add_argument('--loss_clip', type=float, default=5.0)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.pre_epochs < 0 or args.epochs < 0:
        parser.error('epoch counts must be nonnegative')
    if args.pre_epochs == 0 and not args.checkpoint:
        parser.error('--pre_epochs 0 requires --checkpoint (source model weights)')
    if args.batch_size < 2 or args.eval_batch_size < 1 or args.workers < 0:
        parser.error('batch_size >= 2, eval_batch_size >= 1 and workers >= 0 required')
    if args.teacher_temperature <= 0 or args.distribution_ratio_max < 1:
        parser.error('temperature must be positive and distribution_ratio_max >= 1')
    if not 0 <= args.ema_decay < 1 or not 0 <= args.prototype_momentum < 1:
        parser.error('EMA and prototype decay must be in [0, 1)')
    if not 0 <= args.threshold_min <= args.threshold_max <= 1:
        parser.error('require 0 <= threshold_min <= threshold_max <= 1')
    if not 0 <= args.consistency_min_conf <= 1 or not 0 <= args.fallback_conf <= 1:
        parser.error('confidence cutoffs must be in [0, 1]')
    for name in ('lr', 'pseudo_weight_max', 'class_weight_max', 'loss_clip', 'grad_clip'):
        if getattr(args, name) <= 0:
            parser.error('--%s must be positive' % name)
    if args.target_lr is not None and args.target_lr <= 0:
        parser.error('--target_lr must be positive')
    for name in ('w1', 'w2', 'w3', 'lambda_target', 'lambda_aff', 'distribution_power',
                 'prototype_margin', 'target_ramp_epochs', 'aff_warmup_epochs', 'aff_ramp_epochs'):
        if getattr(args, name) < 0:
            parser.error('--%s must be nonnegative' % name)
    return args


def build_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    weak = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    strong = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        normalize,
    ])

    test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return weak, strong, test


def classifier_modulation_loss(model):
    weight = model.fc.weight
    norm = torch.norm(weight, dim=1, keepdim=True).clamp_min(1e-12)
    cosine = weight.mm(weight.t()) / norm.mm(norm.t())
    matrix = cosine - torch.eye(weight.size(0), device=weight.device)
    return torch.mean((matrix + 1.0) / 2.0)


def linear_ramp(epoch, ramp_epochs, maximum=1.0):
    if ramp_epochs <= 0:
        return float(maximum)
    progress = min(float(epoch + 1) / float(ramp_epochs), 1.0)
    return float(maximum) * progress


def affinity_ramp(epoch, warmup_epochs, ramp_epochs, maximum):
    if epoch < warmup_epochs:
        return 0.0
    return linear_ramp(epoch - warmup_epochs, ramp_epochs, maximum)


def calculate_teacher_statistics(teacher, loader, class_num, epoch,
                                 total_epochs, args):
    """Estimate target prior and class-wise thresholds over the full split.

    Two independent weak views are evaluated by the EMA teacher. The averaged
    probabilities are mildly distribution-aligned before per-class thresholds
    are calculated. Ground-truth FER labels are never used here.
    """
    all_probs1, all_probs2 = [], []
    device = next(teacher.parameters()).device
    teacher.eval()
    with torch.no_grad():
        for weak1, weak2, _ in loader:
            weak1 = weak1.to(device, non_blocking=True)
            weak2 = weak2.to(device, non_blocking=True)
            logits1, _ = teacher(weak1, None, None, mode='test', task='target')
            logits2, _ = teacher(weak2, None, None, mode='test', task='target')
            probs1 = F.softmax(logits1 / args.teacher_temperature, dim=1)
            probs2 = F.softmax(logits2 / args.teacher_temperature, dim=1)
            all_probs1.append(probs1.cpu())
            all_probs2.append(probs2.cpu())

    if not all_probs1:
        raise ValueError('Target statistics loader is empty.')
    probs1, probs2 = torch.cat(all_probs1), torch.cat(all_probs2)
    prior = ((probs1 + probs2) * 0.5).mean(dim=0)
    prior = prior / prior.sum().clamp_min(1e-12)

    corrected1, corrected2 = align_dual_view_probabilities(
        probs1, probs2, prior, args.distribution_power, args.distribution_ratio_max
    )
    confidence, predicted = ((corrected1 + corrected2) * 0.5).max(dim=1)

    class_mean = torch.zeros(class_num, dtype=torch.float32)
    class_count = torch.zeros(class_num, dtype=torch.float32)
    for c in range(class_num):
        class_conf = confidence[predicted == c]
        if class_conf.numel() > 0:
            class_mean[c] = class_conf.mean()
            class_count[c] = float(class_conf.numel())

    if total_epochs <= 1:
        progress = 1.0
    else:
        progress = float(epoch) / float(total_epochs - 1)

    thresholds = class_mean * args.phi + args.threshold_stage_gain * progress
    thresholds = torch.clamp(
        thresholds,
        min=args.threshold_min,
        max=args.threshold_max,
    )
    thresholds[class_count == 0] = args.threshold_max
    return thresholds, prior


@torch.no_grad()
def generate_dual_view_pseudo_labels(teacher, weak1, weak2,
                                     thresholds, prior, args,
                                     return_features=False):
    """Frozen weak-view predictions; optionally expose stable teacher features."""
    teacher.eval()
    logits1, features1 = teacher(weak1, mode='test', task='target')
    logits2, features2 = teacher(weak2, mode='test', task='target')
    probs1, probs2 = align_dual_view_probabilities(
        F.softmax(logits1 / args.teacher_temperature, dim=1),
        F.softmax(logits2 / args.teacher_temperature, dim=1),
        prior, args.distribution_power, args.distribution_ratio_max,
    )
    selected = select_dual_view_pseudo_labels(probs1, probs2, thresholds, prior, args)
    if return_features:
        features = F.normalize(
            F.normalize(features1, dim=1) + F.normalize(features2, dim=1), dim=1
        )
        return selected + (features.detach(),)
    return selected


def backward_and_step(loss, model, optimizer, grad_clip):
    """Never write NaN/Inf parameters into the student, teacher or memory."""
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite training loss; optimizer was not stepped.')
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad()
        raise FloatingPointError('Non-finite gradients; optimizer was not stepped.')
    optimizer.step()


def evaluate(model, loader, criterion, num_samples, epoch, split_name):
    val_loss = 0.0
    iter_cnt = 0
    bingo_cnt = 0
    preds, labels = [], []

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            out, _ = model(imgs, targets, None, mode='test')
            loss = torch.mean(criterion(out, targets))
            val_loss += loss.item()
            iter_cnt += 1

            predicts = torch.argmax(out, dim=1)
            bingo_cnt += torch.eq(predicts, targets).sum().item()
            preds.append(predicts.cpu())
            labels.append(targets.cpu())

    avg_loss = val_loss / max(iter_cnt, 1)
    acc = float(bingo_cnt) / float(num_samples)
    print('[Epoch %d] Target %s accuracy: %.4f. Loss: %.3f' %
          (epoch, split_name, acc, avg_loss))
    util.make_confucion_matrix(preds, labels)
    return acc


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc,
                    args, teacher=None, prototypes=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_acc': best_val_acc,
        'args': vars(args),
    }
    if teacher is not None:
        state['ema_teacher'] = teacher.state_dict()
    if prototypes is not None:
        state['prototypes'] = prototypes.state_dict()
    torch.save(state, path)


def run_training():
    args = parse_args()
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    model_path = os.path.join(args.output_dir, args.data1 + '_' + args.data2)
    os.makedirs(model_path, exist_ok=True)

    # Use distinct checkpoint names so this branch cannot overwrite the saved
    # baseline checkpoints produced by sep09-version.
    source_best_path = os.path.join(
        model_path,
        args.backbone + '_' + args.data1 + '_' + args.data2
        + '_ema_dualview_source_best.pth'
    )
    target_best_path = os.path.join(
        model_path,
        args.backbone + '_' + args.data1 + '_' + args.data2
        + '_ema_dualview_target_best.pth'
    )

    print('---------------------------------------------------------------------------------------')
    print('Improved CAST: %s, source %s, target %s' %
          (args.backbone, args.data1, args.data2))
    print('w1:%s w2:%s w3:%s phi:%s ema:%s lambda_t:%s lambda_aff:%s' %
          (args.w1, args.w2, args.w3, args.phi, args.ema_decay,
           args.lambda_target, args.lambda_aff))
    print('---------------------------------------------------------------------------------------')

    if args.backbone not in {'mobilenet_v2', 'resnet18', 'resnet50'}:
        raise ValueError('Backbone Error!')
    train_batch, test_batch = args.batch_size, args.eval_batch_size

    weak_transform, strong_transform, test_transform = build_transforms()

    if args.data1 != 'rafdb' or args.data2 != 'fer':
        raise ValueError('This branch implements RAF-DB -> FER2013.')

    source_train = RafDataSet(
        args.source_path,
        phase='train',
        transform=weak_transform,
        strong_transform=None,
        basic_aug=False,
    )

    # Two weak views plus one strong view for EMA teacher/student consistency.
    target_train = FER(
        args.target_path,
        phase='train',
        transform=weak_transform,
        weak2_transform=weak_transform,
        strong_transform=strong_transform,
        basic_aug=False,
    )
    # Full target-train statistics for global class-wise thresholds and priors.
    target_threshold = FER(
        args.target_path,
        phase='train',
        transform=weak_transform,
        weak2_transform=weak_transform,
        strong_transform=None,
        basic_aug=False,
    )
    target_val = FER(
        args.target_path,
        phase='val',
        transform=test_transform,
        strong_transform=None,
    )
    target_test = FER(
        args.target_path,
        phase='test',
        transform=test_transform,
        strong_transform=None,
    )

    class_num = 7

    if len(source_train) < 2:
        raise ValueError('Source training needs at least two images for BatchNorm.')

    train_loader_source = torch.utils.data.DataLoader(
        source_train, batch_size=train_batch, num_workers=args.workers,
        drop_last=(len(source_train) % train_batch == 1),
        shuffle=True, pin_memory=(device.type == 'cuda'), worker_init_fn=_init_fn,
    )
    train_loader_target = torch.utils.data.DataLoader(
        target_train, batch_size=train_batch, num_workers=args.workers,
        shuffle=True, pin_memory=(device.type == 'cuda'), worker_init_fn=_init_fn,
    )
    threshold_loader_target = torch.utils.data.DataLoader(
        target_threshold, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=(device.type == 'cuda'), worker_init_fn=_init_fn,
    )
    val_loader_target = torch.utils.data.DataLoader(
        target_val, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=(device.type == 'cuda'),
    )
    test_loader_target = torch.utils.data.DataLoader(
        target_test, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=(device.type == 'cuda'),
    )

    model = Networks.Model(
        backbone=args.backbone,
        num_classes=class_num,
        pretrained=not args.no_pretrained and not args.checkpoint,
        class_weight_max=args.class_weight_max,
        loss_clip=args.loss_clip,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if args.checkpoint:
        print('Loading pretrained weights...', args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)

    # ------------------------------------------------------------------
    # Stage 1: source pre-training. Kept close to the baseline for a fair
    # comparison; EMA teacher/prototype innovations begin in target stage.
    # ------------------------------------------------------------------
    best_source_val_acc = -1.0
    for i in range(args.pre_epochs):
        model.train()
        train_cls = 0.0
        train_ddrl = 0.0
        train_weight = 0.0
        batch_count = 0
        bingo_cnt = 0
        sample_count = 0

        for imgs, targets in train_loader_source:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(imgs, targets, None, 'train', 'source')
            cls_loss = torch.mean(criterion(output[0], targets))
            ddrl_loss = output[1]
            weight_loss = classifier_modulation_loss(model)
            loss = (
                cls_loss * args.w1
                + ddrl_loss * args.w2
                + weight_loss * args.w3
            )
            backward_and_step(loss, model, optimizer, args.grad_clip)

            predicts = torch.argmax(output[0], dim=1)
            bingo_cnt += torch.eq(predicts, targets).sum().item()
            sample_count += targets.numel()
            train_cls += cls_loss.item()
            train_ddrl += ddrl_loss.item()
            train_weight += weight_loss.item()
            batch_count += 1

        scheduler.step()
        train_acc = float(bingo_cnt) / max(sample_count, 1)
        print('[Source Epoch %d] Training accuracy: %.4f. Classification Loss: %.3f '
              'Class Enhancement: %.3f Weight Loss: %.3f LR: %.6f' %
              (i, train_acc,
               train_cls / max(batch_count, 1),
               train_ddrl / max(batch_count, 1),
               train_weight / max(batch_count, 1),
               optimizer.param_groups[0]['lr']))

        val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), i, 'Validation'
        )
        if val_acc > best_source_val_acc:
            best_source_val_acc = val_acc
            save_checkpoint(
                source_best_path, model, optimizer, scheduler, i,
                best_source_val_acc, args
            )
            print('best source-stage validation accuracy %.4f' % best_source_val_acc)

    if args.pre_epochs > 0:
        checkpoint = torch.load(source_best_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else:
        # --checkpoint has already loaded the requested source model. Never
        # silently pick up a stale source_best file from an earlier run.
        best_source_val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), -1, 'Validation'
        )
        save_checkpoint(source_best_path, model, optimizer, scheduler, -1,
                        best_source_val_acc, args)

    # A separate optimizer prevents source-stage decay/moments from suppressing
    # adaptation. --target_lr controls this stage independently.
    target_lr = args.lr if args.target_lr is None else args.target_lr
    optimizer = torch.optim.Adam(model.parameters(), target_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # ------------------------------------------------------------------
    # Stage 2: Dual-view EMA teacher + improved DDRL/CCDR.
    # ------------------------------------------------------------------
    teacher = create_ema_teacher(model)
    prototypes = PrototypeMemory(
        num_classes=class_num,
        feature_dim=model.feature_dim,
        device=device,
        momentum=args.prototype_momentum,
        margin=args.prototype_margin,
    )

    best_target_val_acc = -1.0
    global_step = 0

    for i in range(args.epochs):
        thresholds, target_prior = calculate_teacher_statistics(
            teacher,
            threshold_loader_target,
            class_num,
            i,
            args.epochs,
            args,
        )
        print('[Target Epoch %d] class-wise thresholds: %s' %
              (i, np.array2string(thresholds.numpy(), precision=4)))
        print('[Target Epoch %d] corrected target prior: %s' %
              (i, np.array2string(target_prior.numpy(), precision=4)))

        target_weight = linear_ramp(
            i, args.target_ramp_epochs, args.lambda_target
        )
        proto_weight = affinity_ramp(
            i, args.aff_warmup_epochs, args.aff_ramp_epochs, args.lambda_aff
        )

        model.train()
        source_train_iter = iter(train_loader_source)
        stats = {
            'source_cls': 0.0,
            'target_cls': 0.0,
            'ddrl': 0.0,
            'domain': 0.0,
            'class_sep': 0.0,
            'proto': 0.0,
            'weight': 0.0,
        }
        batch_count = 0
        confident_num = 0
        agreement_num = 0
        pseudo_distribution = np.zeros(class_num, dtype=np.int64)

        for weak1, weak2, strong, _ in train_loader_target:
            try:
                source_imgs, source_targets = next(source_train_iter)
            except StopIteration:
                source_train_iter = iter(train_loader_source)
                source_imgs, source_targets = next(source_train_iter)

            weak1 = weak1.to(device, non_blocking=True)
            weak2 = weak2.to(device, non_blocking=True)
            strong = strong.to(device, non_blocking=True)
            source_imgs = source_imgs.to(device, non_blocking=True)
            source_targets = source_targets.to(device, non_blocking=True)

            pseudo_targets, target_mask, pseudo_weights, agree_count, teacher_target_features = (
                generate_dual_view_pseudo_labels(
                    teacher, weak1, weak2, thresholds, target_prior, args,
                    return_features=True,
                )
            )
            agreement_num += agree_count
            confident_num += int(target_mask.sum().item())

            reliable_cpu = pseudo_targets[target_mask.bool()].detach().cpu().numpy()
            for c in range(class_num):
                pseudo_distribution[c] += int(np.sum(reliable_cpu == c))

            source_mask = torch.ones(
                source_imgs.size(0), dtype=target_mask.dtype, device=device
            )
            train_imgs = torch.cat((source_imgs, strong), dim=0)
            train_targets = torch.cat((source_targets, pseudo_targets), dim=0)
            train_mask = torch.cat((source_mask, target_mask), dim=0)

            optimizer.zero_grad()
            output = model(
                train_imgs,
                train_targets,
                train_mask,
                'train',
                'target',
                source_count=source_imgs.size(0),
            )

            source_count = source_imgs.size(0)
            source_logits = output[0][:source_count]
            target_logits = output[0][source_count:]
            source_cls_loss = criterion(source_logits, source_targets).mean()

            target_cls_loss = weighted_mean_loss(
                criterion(target_logits, pseudo_targets), pseudo_weights * target_mask
            )

            cls_loss = source_cls_loss + target_weight * target_cls_loss
            ddrl_loss = output[1]
            domain_loss = output[3]
            class_sep_loss = output[4]
            weight_loss = classifier_modulation_loss(model)

            all_features = output[2]
            target_features = all_features[source_count:]

            # Use eval-mode EMA features (no dropout / strong-view noise) for
            # stable memory anchors; live student features receive affinity gradients.
            with torch.no_grad():
                _, teacher_source_features = teacher(source_imgs, mode='test')
            prototypes.update(teacher_source_features, source_targets)
            reliable = target_mask.bool()
            if torch.any(reliable):
                proto_loss = prototypes.loss(
                    target_features[reliable],
                    pseudo_targets[reliable],
                    pseudo_weights[reliable],
                )
            else:
                proto_loss = target_features.sum() * 0.0

            loss = (
                args.w1 * cls_loss
                + args.w2 * ddrl_loss
                + proto_weight * proto_loss
                + args.w3 * weight_loss
            )
            backward_and_step(loss, model, optimizer, args.grad_clip)

            # Only reliable weak-view teacher features enter target memory.
            if torch.any(reliable):
                prototypes.update(
                    teacher_target_features[reliable],
                    pseudo_targets[reliable],
                    pseudo_weights[reliable],
                )

            global_step += 1
            update_ema_teacher(teacher, model, args.ema_decay, global_step)

            stats['source_cls'] += source_cls_loss.item()
            stats['target_cls'] += target_cls_loss.item()
            stats['ddrl'] += ddrl_loss.item()
            stats['domain'] += domain_loss.item()
            stats['class_sep'] += class_sep_loss.item()
            stats['proto'] += proto_loss.item()
            stats['weight'] += weight_loss.item()
            batch_count += 1

        scheduler.step()
        denominator = max(batch_count, 1)
        print('[Target Epoch %d] Agreement_Num: %d Confident_Num: %d '
              'Pseudo_Distribution: %s' %
              (i, agreement_num, confident_num,
               np.array2string(pseudo_distribution, separator=', ')))
        print('[Target Epoch %d] SourceCE: %.3f TargetCE: %.3f lambda_t: %.3f '
              'DDRL: %.3f (Domain: %.3f ClassSep: %.3f) ProtoAff: %.3f '
              'lambda_aff: %.3f Classifier: %.3f LR: %.6f' %
              (i,
               stats['source_cls'] / denominator,
               stats['target_cls'] / denominator,
               target_weight,
               stats['ddrl'] / denominator,
               stats['domain'] / denominator,
               stats['class_sep'] / denominator,
               stats['proto'] / denominator,
               proto_weight,
               stats['weight'] / denominator,
               optimizer.param_groups[0]['lr']))

        val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), i, 'Validation'
        )
        if val_acc > best_target_val_acc:
            best_target_val_acc = val_acc
            save_checkpoint(
                target_best_path,
                model,
                optimizer,
                scheduler,
                i,
                best_target_val_acc,
                args,
                teacher=teacher,
                prototypes=prototypes,
            )
            print('best target-stage validation accuracy %.4f' % best_target_val_acc)

    if args.epochs > 0:
        checkpoint = torch.load(target_best_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        selected_val_acc = best_target_val_acc
    else:
        checkpoint = torch.load(source_best_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        selected_val_acc = best_source_val_acc

    test_acc = evaluate(
        model, test_loader_target, criterion, len(target_test), args.epochs, 'Test'
    )
    print('best source-stage validation accuracy %.4f' % best_source_val_acc)
    if args.epochs > 0:
        print('best target-stage validation accuracy %.4f' % best_target_val_acc)
    print('selected validation accuracy %.4f' % selected_val_acc)
    print('final target test accuracy %.4f' % test_acc)
    return test_acc


if __name__ == '__main__':
    run_training()
