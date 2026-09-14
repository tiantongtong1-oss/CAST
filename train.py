import warnings
warnings.filterwarnings('ignore')

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
    distribution_align,
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


def parse_args():
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
    return parser.parse_args()


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
    all_probs = []
    teacher.eval()
    with torch.no_grad():
        for weak1, weak2, _ in loader:
            weak1 = weak1.cuda(non_blocking=True)
            weak2 = weak2.cuda(non_blocking=True)
            logits1, _ = teacher(weak1, None, None, mode='test', task='target')
            logits2, _ = teacher(weak2, None, None, mode='test', task='target')
            probs1 = F.softmax(logits1 / args.teacher_temperature, dim=1)
            probs2 = F.softmax(logits2 / args.teacher_temperature, dim=1)
            all_probs.append(((probs1 + probs2) * 0.5).cpu())

    probs = torch.cat(all_probs, dim=0)
    prior = probs.mean(dim=0)
    prior = prior / prior.sum().clamp_min(1e-12)

    corrected = distribution_align(
        probs,
        prior,
        power=args.distribution_power,
        ratio_max=args.distribution_ratio_max,
    )
    confidence, predicted = corrected.max(dim=1)

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


def generate_dual_view_pseudo_labels(teacher, weak1, weak2,
                                     thresholds, prior, args):
    """EMA pseudo labels with two-view agreement and confidence filtering."""
    teacher.eval()
    with torch.no_grad():
        logits1, _ = teacher(weak1, None, None, mode='test', task='target')
        logits2, _ = teacher(weak2, None, None, mode='test', task='target')

        probs1 = F.softmax(logits1 / args.teacher_temperature, dim=1)
        probs2 = F.softmax(logits2 / args.teacher_temperature, dim=1)
        probs1 = distribution_align(
            probs1, prior, args.distribution_power, args.distribution_ratio_max
        )
        probs2 = distribution_align(
            probs2, prior, args.distribution_power, args.distribution_ratio_max
        )

        conf1, pred1 = probs1.max(dim=1)
        conf2, pred2 = probs2.max(dim=1)
        mean_probs = (probs1 + probs2) * 0.5
        confidence, pseudo_targets = mean_probs.max(dim=1)

        agreement = pred1.eq(pred2) & pred1.eq(pseudo_targets)
        min_view_conf = torch.minimum(conf1, conf2)
        sample_threshold = thresholds.to(weak1.device).index_select(0, pseudo_targets)

        reliable = (
            agreement
            & (confidence >= sample_threshold)
            & (min_view_conf >= args.consistency_min_conf)
        )

        # High-confidence fallback prevents an empty target batch while still
        # requiring both teacher views to agree.
        if not torch.any(reliable):
            fallback = agreement & (confidence >= args.fallback_conf)
            if torch.any(fallback):
                fallback_indices = fallback.nonzero(as_tuple=False).squeeze(1)
                best_local = confidence.index_select(0, fallback_indices).argmax()
                reliable[fallback_indices[best_local]] = True

        # Reliability and mild inverse-prior weighting are both bounded.
        uniform = torch.full_like(prior, 1.0 / float(prior.numel()))
        balance = torch.pow(
            uniform.to(weak1.device)
            / prior.to(weak1.device).clamp_min(1e-6),
            args.distribution_power * 0.5,
        )
        balance = torch.clamp(
            balance,
            min=1.0 / args.distribution_ratio_max,
            max=args.distribution_ratio_max,
        )
        class_weight = balance.index_select(0, pseudo_targets)
        pseudo_weight = torch.sqrt(
            (confidence * min_view_conf).clamp_min(0.0)
        ) * class_weight
        pseudo_weight = torch.clamp(
            pseudo_weight, min=0.0, max=args.pseudo_weight_max
        )
        pseudo_weight = pseudo_weight * reliable.float()

    return pseudo_targets, reliable.float(), pseudo_weight, int(agreement.sum().item())


def evaluate(model, loader, criterion, num_samples, epoch, split_name):
    val_loss = 0.0
    iter_cnt = 0
    bingo_cnt = 0
    preds, labels = [], []

    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
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
    model_path = os.path.join('./models', args.data1 + '_' + args.data2)
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

    if args.backbone == 'resnet18':
        train_batch, test_batch = 128, 128
    elif args.backbone == 'resnet50':
        train_batch, test_batch = 128, 100
    elif args.backbone == 'mobilenet_v2':
        train_batch, test_batch = 128, 128
    else:
        raise ValueError('Backbone Error!')

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

    train_loader_source = torch.utils.data.DataLoader(
        source_train, batch_size=train_batch, num_workers=args.workers,
        shuffle=True, pin_memory=True, worker_init_fn=_init_fn,
    )
    train_loader_target = torch.utils.data.DataLoader(
        target_train, batch_size=train_batch, num_workers=args.workers,
        shuffle=True, pin_memory=True, worker_init_fn=_init_fn,
    )
    threshold_loader_target = torch.utils.data.DataLoader(
        target_threshold, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True, worker_init_fn=_init_fn,
    )
    val_loader_target = torch.utils.data.DataLoader(
        target_val, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True,
    )
    test_loader_target = torch.utils.data.DataLoader(
        target_test, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True,
    )

    model = Networks.Model(
        backbone=args.backbone,
        num_classes=class_num,
        class_weight_max=args.class_weight_max,
        loss_clip=args.loss_clip,
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if args.checkpoint:
        print('Loading pretrained weights...', args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cuda')
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
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

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

    checkpoint = torch.load(source_best_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    # ------------------------------------------------------------------
    # Stage 2: Dual-view EMA teacher + improved DDRL/CCDR.
    # ------------------------------------------------------------------
    teacher = create_ema_teacher(model)
    prototypes = PrototypeMemory(
        num_classes=class_num,
        feature_dim=model.feature_dim,
        device=torch.device('cuda'),
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

            weak1 = weak1.cuda(non_blocking=True)
            weak2 = weak2.cuda(non_blocking=True)
            strong = strong.cuda(non_blocking=True)
            source_imgs = source_imgs.cuda(non_blocking=True)
            source_targets = source_targets.cuda(non_blocking=True)

            pseudo_targets, target_mask, pseudo_weights, agree_count = (
                generate_dual_view_pseudo_labels(
                    teacher, weak1, weak2, thresholds, target_prior, args
                )
            )
            agreement_num += agree_count
            confident_num += int(target_mask.sum().item())

            reliable_cpu = pseudo_targets[target_mask.bool()].detach().cpu().numpy()
            for c in range(class_num):
                pseudo_distribution[c] += int(np.sum(reliable_cpu == c))

            source_mask = torch.ones(
                source_imgs.size(0), dtype=target_mask.dtype, device='cuda'
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

            if torch.any(target_mask > 0):
                target_ce = criterion(target_logits, pseudo_targets)
                effective_weight = pseudo_weights * target_mask
                target_cls_loss = (
                    (target_ce * effective_weight).sum()
                    / effective_weight.sum().clamp_min(1e-6)
                )
            else:
                target_cls_loss = target_logits.sum() * 0.0

            cls_loss = source_cls_loss + target_weight * target_cls_loss
            ddrl_loss = output[1]
            domain_loss = output[3]
            class_sep_loss = output[4]
            weight_loss = classifier_modulation_loss(model)

            all_features = output[2]
            source_features = all_features[:source_count]
            target_features = all_features[source_count:]

            # Source truth anchors prototypes; target samples contribute only
            # if both EMA views agree and pass the class-wise threshold.
            prototypes.update(source_features, source_targets)
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
            loss = torch.nan_to_num(
                loss, nan=0.0, posinf=args.loss_clip, neginf=-args.loss_clip
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Update target prototype memory only after the student step, using
            # detached reliable features so pseudo-label noise cannot backprop.
            if torch.any(reliable):
                prototypes.update(
                    target_features[reliable],
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
        checkpoint = torch.load(target_best_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        selected_val_acc = best_target_val_acc
    else:
        checkpoint = torch.load(source_best_path, map_location='cuda')
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
