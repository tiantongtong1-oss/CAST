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
from ema_utils import create_ema_teacher, select_dual_view_pseudo_labels, update_ema_teacher
from prototype_utils import FeatureHook, PrototypeBank, prototype_weight_for_epoch
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
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        help='mobilenet_v2, resnet18 or resnet50')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument('--pre_epochs', type=int, default=30,
                        help='source-domain pre-training epochs')
    parser.add_argument('--epochs', type=int, default=30,
                        help='target-domain self-training epochs')
    parser.add_argument('--w1', type=float, default=4.0,
                        help='alpha: classification loss weight')
    parser.add_argument('--w2', type=float, default=0.3,
                        help='beta: feature modulation loss weight')
    parser.add_argument('--w3', type=float, default=0.1,
                        help='gamma: classifier modulation loss weight')
    parser.add_argument('--phi', type=float, default=1.4,
                        help='legacy CAST threshold arg retained for CLI compatibility; '
                             'not used by the stable relative threshold')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='fixed EMA decay for teacher parameters and floating buffers')
    parser.add_argument('--threshold_base', type=float, default=0.85,
                        help='minimum global center of the adaptive threshold')
    parser.add_argument('--threshold_beta', type=float, default=0.5,
                        help='strength of per-class confidence deviation')
    parser.add_argument('--threshold_margin', type=float, default=0.02,
                        help='margin added above teacher global mean confidence')
    parser.add_argument('--threshold_min', type=float, default=0.80,
                        help='minimum class-adaptive threshold')
    parser.add_argument('--threshold_max', type=float, default=0.95,
                        help='maximum class-adaptive threshold')

    # Prototype-consistency v1. The defaults intentionally keep this
    # regularizer weaker than the main CAST classification objective.
    parser.add_argument('--proto_weight', type=float, default=0.10,
                        help='maximum prototype-consistency loss weight')
    parser.add_argument('--proto_temperature', type=float, default=0.20,
                        help='temperature for prototype cosine logits')
    parser.add_argument('--proto_momentum', type=float, default=0.99,
                        help='EMA momentum of target class prototypes')
    parser.add_argument('--proto_source_anchor', type=float, default=0.50,
                        help='source-prototype fraction in the blended prototype')
    parser.add_argument('--proto_warmup_epochs', type=int, default=3,
                        help='epochs that only update prototype memory, without its loss')
    parser.add_argument('--proto_ramp_epochs', type=int, default=5,
                        help='epochs used to linearly ramp prototype loss to full weight')

    args = parser.parse_args()
    if not (0.0 <= args.threshold_min <= args.threshold_max <= 1.0):
        parser.error('require 0 <= threshold_min <= threshold_max <= 1')
    if not (0.0 <= args.threshold_base <= args.threshold_max):
        parser.error('require 0 <= threshold_base <= threshold_max')
    if args.threshold_beta < 0.0:
        parser.error('--threshold_beta must be nonnegative')
    if args.threshold_margin < 0.0:
        parser.error('--threshold_margin must be nonnegative')
    if not (0.0 <= args.ema_decay < 1.0):
        parser.error('--ema_decay must be in [0, 1)')
    if args.proto_weight < 0.0:
        parser.error('--proto_weight must be nonnegative')
    if args.proto_temperature <= 0.0:
        parser.error('--proto_temperature must be positive')
    if not (0.0 <= args.proto_momentum < 1.0):
        parser.error('--proto_momentum must be in [0, 1)')
    if not (0.0 <= args.proto_source_anchor <= 1.0):
        parser.error('--proto_source_anchor must be in [0, 1]')
    if args.proto_warmup_epochs < 0 or args.proto_ramp_epochs < 0:
        parser.error('prototype warmup/ramp epochs must be nonnegative')
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


def calculate_target_thresholds(model, loader, class_num, threshold_base,
                                threshold_beta, threshold_margin,
                                threshold_min, threshold_max):
    """Global-confidence-aware relative class-adaptive thresholds."""
    class_sum = torch.zeros(class_num, dtype=torch.float64)
    class_count = torch.zeros(class_num, dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].cuda(non_blocking=True)
            logits, _ = model(imgs, None, None, mode='test', task='target')
            probs = F.softmax(logits, dim=1).cpu()
            max_prob, pred_label = torch.max(probs, dim=1)

            class_sum.scatter_add_(0, pred_label, max_prob.double())
            class_count.scatter_add_(
                0, pred_label, torch.ones_like(max_prob, dtype=torch.float64)
            )

    valid = class_count > 0
    class_mean = class_sum / class_count.clamp_min(1.0)
    if torch.any(valid):
        global_mean = class_mean[valid].mean()
    else:
        global_mean = class_mean.new_tensor(float(threshold_base))

    threshold_center = max(
        float(threshold_base),
        float(global_mean.item()) + float(threshold_margin),
    )
    threshold_center = min(threshold_center, float(threshold_max))

    thresholds = threshold_center + threshold_beta * (class_mean - global_mean)
    thresholds = torch.clamp(
        thresholds, min=threshold_min, max=threshold_max
    ).float()
    thresholds[~valid] = float(threshold_max)
    return (
        thresholds,
        class_mean.float(),
        float(global_mean.item()),
        float(threshold_center),
    )


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


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, args,
                    teacher=None, prototype_bank=None):
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
    if prototype_bank is not None:
        state['prototype_bank'] = prototype_bank.state_dict()
    torch.save(state, path)


def initialize_source_prototypes(teacher, feature_hook, loader, prototype_bank):
    """Build deterministic source semantic anchors from labeled RAF-DB."""
    teacher.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            teacher(imgs, None, None, mode='test', task='source')
            prototype_bank.accumulate_source(feature_hook.output, targets)
    prototype_bank.finalize_source()


def run_training():
    args = parse_args()
    model_path = os.path.join('./models', args.data1 + '_' + args.data2)
    os.makedirs(model_path, exist_ok=True)

    source_best_path = os.path.join(
        model_path,
        args.backbone + '_' + args.data1 + '_' + args.data2
        + '_prototype_consistency_v1_source_best.pth'
    )
    target_best_path = os.path.join(
        model_path,
        args.backbone + '_' + args.data1 + '_' + args.data2
        + '_prototype_consistency_v1_target_best.pth'
    )

    print('---------------------------------------------------------------------------------------')
    print('EMA + Dual View + Stable CAT + Prototype Consistency v1: %s with source %s and target %s' %
          (args.backbone, args.data1, args.data2))
    print('alpha(w1):%s beta(w2):%s gamma(w3):%s ema:%s '
          'tau0:%s threshold_beta:%s margin:%s threshold_range:[%s,%s]' %
          (args.w1, args.w2, args.w3, args.ema_decay,
           args.threshold_base, args.threshold_beta, args.threshold_margin,
           args.threshold_min, args.threshold_max))
    print('prototype weight:%s temp:%s momentum:%s source_anchor:%s warmup:%s ramp:%s' %
          (args.proto_weight, args.proto_temperature, args.proto_momentum,
           args.proto_source_anchor, args.proto_warmup_epochs, args.proto_ramp_epochs))
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
        args.source_path, phase='train', transform=weak_transform,
        strong_transform=None, basic_aug=False
    )
    # Deterministic source view for semantic prototype initialization.
    source_prototype = RafDataSet(
        args.source_path, phase='train', transform=test_transform,
        strong_transform=None, basic_aug=False
    )

    target_train = FER(
        args.target_path, phase='train', transform=weak_transform,
        weak2_transform=weak_transform, strong_transform=strong_transform,
        basic_aug=False
    )
    target_threshold = FER(
        args.target_path, phase='train', transform=weak_transform,
        strong_transform=None, basic_aug=False
    )
    target_val = FER(
        args.target_path, phase='val', transform=test_transform,
        strong_transform=None
    )
    target_test = FER(
        args.target_path, phase='test', transform=test_transform,
        strong_transform=None
    )

    class_num = 7

    train_loader_source = torch.utils.data.DataLoader(
        source_train, batch_size=train_batch, num_workers=args.workers,
        shuffle=True, pin_memory=True, worker_init_fn=_init_fn,
    )
    prototype_loader_source = torch.utils.data.DataLoader(
        source_prototype, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True, worker_init_fn=_init_fn,
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

    model = Networks.Model(backbone=args.backbone, num_classes=class_num).cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if args.checkpoint:
        print('Loading pretrained weights...', args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cuda')
        model.load_state_dict(checkpoint['model'], strict=True)

    best_source_val_acc = -1.0
    for i in range(args.pre_epochs):
        model.train()
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        batch_count = 0
        bingo_cnt = 0
        sample_count = 0

        for imgs, targets in train_loader_source:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            optimizer.zero_grad()
            output = model(imgs, targets, None, 'train', 'source')
            cls_loss = torch.mean(criterion(output[0], targets))
            aff_loss = output[1]
            weight_loss = classifier_modulation_loss(model)
            loss = cls_loss * args.w1 + aff_loss * args.w2 + weight_loss * args.w3
            loss.backward()
            optimizer.step()

            predicts = torch.argmax(output[0], dim=1)
            bingo_cnt += torch.eq(predicts, targets).sum().item()
            sample_count += targets.numel()
            train_loss1 += cls_loss.item()
            train_loss2 += aff_loss.item()
            train_loss3 += weight_loss.item()
            batch_count += 1

        scheduler.step()
        train_acc = float(bingo_cnt) / max(sample_count, 1)
        print('[Source Epoch %d] Training accuracy: %.4f. Classification Loss: %.3f '
              'Affinity Loss: %.3f Weight Loss: %.3f LR: %.6f' %
              (i, train_acc,
               train_loss1 / max(batch_count, 1),
               train_loss2 / max(batch_count, 1),
               train_loss3 / max(batch_count, 1),
               optimizer.param_groups[0]['lr']))

        val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), i, 'Validation'
        )
        if val_acc > best_source_val_acc:
            best_source_val_acc = val_acc
            save_checkpoint(source_best_path, model, optimizer, scheduler, i,
                            best_source_val_acc, args)
            print('best source-stage validation accuracy %.4f' % best_source_val_acc)

    checkpoint = torch.load(source_best_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    teacher = create_ema_teacher(model).cuda()
    teacher_feature_hook = FeatureHook(teacher.feature)
    student_feature_hook = FeatureHook(model.feature)

    prototype_bank = PrototypeBank(
        class_num,
        model.fc.in_features,
        momentum=args.proto_momentum,
        source_anchor=args.proto_source_anchor,
    ).cuda()
    initialize_source_prototypes(
        teacher, teacher_feature_hook, prototype_loader_source, prototype_bank
    )
    print('source prototype counts: %s' %
          np.array2string(prototype_bank.source_counts.cpu().numpy(), separator=', '))

    best_target_val_acc = -1.0
    global_step = 0

    for i in range(args.epochs):
        thresholds, class_mean, global_mean, threshold_center = calculate_target_thresholds(
            teacher,
            threshold_loader_target,
            class_num,
            args.threshold_base,
            args.threshold_beta,
            args.threshold_margin,
            args.threshold_min,
            args.threshold_max,
        )
        current_proto_weight = prototype_weight_for_epoch(
            i, args.proto_weight, args.proto_warmup_epochs, args.proto_ramp_epochs
        )

        print('[Target Epoch %d] class mean confidence: %s global_mean: %.4f '
              'threshold_center: %.4f' %
              (i, np.array2string(class_mean.numpy(), precision=4),
               global_mean, threshold_center))
        print('[Target Epoch %d] class-adaptive thresholds: %s' %
              (i, np.array2string(thresholds.numpy(), precision=4)))

        model.train()
        source_train_iter = iter(train_loader_source)
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        train_proto_loss = 0.0
        batch_count = 0
        confident_num = 0
        agreement_num = 0
        prototype_agree_num = 0
        prototype_checked_num = 0
        prototype_similarity_sum = 0.0
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

            teacher.eval()
            with torch.no_grad():
                logits1, _ = teacher(weak1, None, None, 'test', 'target')
                teacher_features1 = teacher_feature_hook.output.detach()
                logits2, _ = teacher(weak2, None, None, 'test', 'target')
                teacher_features2 = teacher_feature_hook.output.detach()
                pseudo_targets, con_idx, agree_count = select_dual_view_pseudo_labels(
                    logits1, logits2, thresholds
                )
                teacher_mean_features = F.normalize(
                    F.normalize(teacher_features1, dim=1)
                    + F.normalize(teacher_features2, dim=1),
                    dim=1,
                )
                proto_agree, proto_checked, proto_similarity = (
                    prototype_bank.agreement_stats(
                        teacher_mean_features, pseudo_targets, con_idx
                    )
                )

            agreement_num += agree_count
            confident_num += int(con_idx.sum().item())
            prototype_agree_num += proto_agree
            prototype_checked_num += proto_checked
            prototype_similarity_sum += proto_similarity * proto_checked
            reliable = pseudo_targets[con_idx.bool()].cpu().numpy()
            for c in range(class_num):
                pseudo_distribution[c] += int(np.sum(reliable == c))

            model.train()
            source_con_idx = torch.ones(source_imgs.shape[0])
            train_imgs = torch.cat((source_imgs, strong.cpu()), dim=0).cuda(non_blocking=True)
            train_targets = torch.cat((source_targets, pseudo_targets.cpu()), dim=0).cuda(non_blocking=True)
            train_con_idx = torch.cat((source_con_idx, con_idx.cpu()), dim=0).cuda(non_blocking=True)

            optimizer.zero_grad()
            output = model(
                train_imgs, train_targets, train_con_idx, 'train', 'target',
                source_count=source_imgs.shape[0],
            )

            per_sample_loss = criterion(output[0], train_targets) * train_con_idx
            cls_loss = per_sample_loss.sum() / train_con_idx.sum().clamp_min(1.0)
            aff_loss = output[1]
            weight_loss = classifier_modulation_loss(model)

            target_student_features = student_feature_hook.output[source_imgs.shape[0]:]
            proto_loss = prototype_bank.consistency_loss(
                target_student_features,
                pseudo_targets,
                con_idx,
                temperature=args.proto_temperature,
            )

            loss = (
                cls_loss * args.w1
                + aff_loss * args.w2
                + weight_loss * args.w3
                + proto_loss * current_proto_weight
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            update_ema_teacher(teacher, model, args.ema_decay, global_step)
            prototype_bank.update_target(
                teacher_mean_features, pseudo_targets, con_idx
            )

            train_loss1 += cls_loss.item()
            train_loss2 += aff_loss.item()
            train_loss3 += weight_loss.item()
            train_proto_loss += proto_loss.item()
            batch_count += 1

        scheduler.step()
        proto_agreement_rate = (
            float(prototype_agree_num) / float(prototype_checked_num)
            if prototype_checked_num > 0 else 0.0
        )
        proto_mean_similarity = (
            prototype_similarity_sum / float(prototype_checked_num)
            if prototype_checked_num > 0 else 0.0
        )
        print('[Target Epoch %d] Agreement_Num: %d Confident_Num: %d '
              'Pseudo_Distribution: %s' %
              (i, agreement_num, confident_num,
               np.array2string(pseudo_distribution, separator=', ')))
        print('[Target Epoch %d] Prototype_Agreement: %d/%d (%.4f) '
              'Mean_Assigned_Cosine: %.4f Target_Prototype_Counts: %s' %
              (i, prototype_agree_num, prototype_checked_num,
               proto_agreement_rate, proto_mean_similarity,
               np.array2string(prototype_bank.target_counts.cpu().numpy(), separator=', ')))
        print('[Target Epoch %d] Classification Loss: %.3f Affinity Loss: %.3f '
              'Weight Loss: %.3f Prototype Loss: %.3f ProtoWeight: %.4f LR: %.6f' %
              (i,
               train_loss1 / max(batch_count, 1),
               train_loss2 / max(batch_count, 1),
               train_loss3 / max(batch_count, 1),
               train_proto_loss / max(batch_count, 1),
               current_proto_weight,
               optimizer.param_groups[0]['lr']))

        val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), i, 'Validation'
        )
        if val_acc > best_target_val_acc:
            best_target_val_acc = val_acc
            save_checkpoint(
                target_best_path, model, optimizer, scheduler, i,
                best_target_val_acc, args, teacher=teacher,
                prototype_bank=prototype_bank,
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

    teacher_feature_hook.close()
    student_feature_hook.close()

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
