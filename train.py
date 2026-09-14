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
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='resnet18, resnet50 or mobilenet_v2')
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
                        help='class-adaptive threshold coefficient')
    return parser.parse_args()


def build_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Paper protocol: faces are aligned/cropped to 256x256, then training uses
    # random 224x224 crops, flipping and rotation. Strong target views follow
    # FixMatch-style RandAugment consistency learning.
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
    """Eq. (6): enlarge angular discrepancies between class classifiers."""
    weight = model.fc.weight
    norm = torch.norm(weight, dim=1, keepdim=True).clamp_min(1e-12)
    cosine = weight.mm(weight.t()) / norm.mm(norm.t())
    matrix = cosine - torch.eye(weight.size(0), device=weight.device)
    return torch.mean((matrix + 1.0) / 2.0)


def calculate_target_thresholds(model, loader, class_num, epoch, total_epochs, phi):
    """Calculate Eq. (7)-(8) over the whole target training split.

    This intentionally does not use target ground-truth labels. Only model
    probabilities on weakly augmented target images contribute to thresholds.
    """
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
            class_count.scatter_add_(0, pred_label, torch.ones_like(max_prob, dtype=torch.float64))

    class_mean = class_sum / class_count.clamp_min(1.0)
    base_threshold = class_mean * phi
    stage_factor = float(total_epochs) / float(total_epochs - epoch)
    thresholds = torch.clamp(base_threshold * stage_factor, max=0.9).float()

    # Eq. (7) is undefined for a class with zero predicted samples. Be
    # conservative in that rare case rather than accepting arbitrary samples.
    thresholds[class_count == 0] = 0.9
    return thresholds


def annotate_target(logits, thresholds):
    probs = F.softmax(logits.detach().cpu(), dim=1)
    pred_values, pred_targets = torch.max(probs, dim=1)
    sample_threshold = thresholds.index_select(0, pred_targets)
    confidence_mask = (pred_values >= sample_threshold).float()

    confident_targets = pred_targets[confidence_mask.bool()].numpy()
    label_dis = [np.sum(confident_targets == i) for i in range(len(thresholds))]
    return pred_targets, confidence_mask, label_dis


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


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, args):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_acc': best_val_acc,
        'args': vars(args),
    }, path)


def run_training():
    args = parse_args()
    model_path = os.path.join('./models', args.data1 + '_' + args.data2)
    os.makedirs(model_path, exist_ok=True)

    source_best_path = os.path.join(
        model_path,
        args.backbone + '_' + args.data1 + '_' + args.data2 + '_source_best.pth'
    )
    target_best_path = os.path.join(
        model_path,
        args.backbone + '_' + args.data1 + '_' + args.data2 + '_target_best.pth'
    )

    print('---------------------------------------------------------------------------------------')
    print('Training %s with source data %s and target data %s' %
          (args.backbone, args.data1, args.data2))
    print('alpha(w1):%s beta(w2):%s gamma(w3):%s phi:%s' %
          (args.w1, args.w2, args.w3, args.phi))
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

    if args.data1 != 'rafdb':
        raise ValueError('This branch currently implements the paper RAF-DB -> FER2013 experiment.')
    if args.data2 != 'fer':
        raise ValueError('This branch currently implements the paper RAF-DB -> FER2013 experiment.')

    source_train = RafDataSet(
        args.source_path,
        phase='train',
        transform=weak_transform,
        strong_transform=None,
        basic_aug=False
    )

    # Target training labels are loaded by Dataset only for evaluation/debugging;
    # they are never used by the optimization or threshold calculation.
    target_train = FER(
        args.target_path,
        phase='train',
        transform=weak_transform,
        strong_transform=strong_transform,
        basic_aug=False
    )
    target_threshold = FER(
        args.target_path,
        phase='train',
        transform=weak_transform,
        strong_transform=None,
        basic_aug=False
    )
    target_val = FER(
        args.target_path,
        phase='val',
        transform=test_transform,
        strong_transform=None
    )
    target_test = FER(
        args.target_path,
        phase='test',
        transform=test_transform,
        strong_transform=None
    )

    class_num = 7

    train_loader_source = torch.utils.data.DataLoader(
        source_train,
        batch_size=train_batch,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )
    train_loader_target = torch.utils.data.DataLoader(
        target_train,
        batch_size=train_batch,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )
    threshold_loader_target = torch.utils.data.DataLoader(
        target_threshold,
        batch_size=test_batch,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )
    val_loader_target = torch.utils.data.DataLoader(
        target_val,
        batch_size=test_batch,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
    )
    test_loader_target = torch.utils.data.DataLoader(
        target_test,
        batch_size=test_batch,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
    )

    model = Networks.Model(backbone=args.backbone, num_classes=class_num)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if args.checkpoint:
        print('Loading pretrained weights...', args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cuda')
        model.load_state_dict(checkpoint['model'], strict=True)

    # ----------------------------------------------------------------------
    # Stage 1: source-domain pre-training (30 epochs in the paper).
    # ----------------------------------------------------------------------
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
            save_checkpoint(
                source_best_path, model, optimizer, scheduler, i,
                best_source_val_acc, args
            )
            print('best source-stage validation accuracy %.4f' % best_source_val_acc)

    # Transfer starts from the best source-pretrained checkpoint selected on
    # the target validation split, never from the target test split.
    checkpoint = torch.load(source_best_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    # ----------------------------------------------------------------------
    # Stage 2: target-domain class-adaptive self-training (30 epochs).
    # Use an independent best metric/checkpoint for this stage so a strong
    # source-only validation score cannot prevent adapted models from saving.
    # ----------------------------------------------------------------------
    best_target_val_acc = -1.0
    for i in range(args.epochs):
        thresholds = calculate_target_thresholds(
            model,
            threshold_loader_target,
            class_num,
            i,
            args.epochs,
            args.phi,
        )
        print('[Target Epoch %d] class-adaptive thresholds: %s' %
              (i, np.array2string(thresholds.numpy(), precision=4)))

        model.train()
        source_train_iter = iter(train_loader_source)
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        batch_count = 0
        confident_num = 0

        for imgs, imgs_aug, _ in train_loader_target:
            try:
                source_imgs, source_targets = next(source_train_iter)
            except StopIteration:
                source_train_iter = iter(train_loader_source)
                source_imgs, source_targets = next(source_train_iter)

            model.eval()
            with torch.no_grad():
                out, _ = model(imgs.cuda(non_blocking=True), None, None, 'test', 'target')
            pseudo_targets, con_idx, _ = annotate_target(out, thresholds)

            model.train()
            source_con_idx = torch.ones(source_imgs.shape[0])
            train_imgs = torch.cat((source_imgs, imgs_aug), dim=0).cuda(non_blocking=True)
            train_targets = torch.cat((source_targets, pseudo_targets), dim=0).cuda(non_blocking=True)
            train_con_idx = torch.cat((source_con_idx, con_idx), dim=0).cuda(non_blocking=True)

            optimizer.zero_grad()
            output = model(
                train_imgs,
                train_targets,
                train_con_idx,
                'train',
                'target',
                source_count=source_imgs.shape[0],
            )

            per_sample_loss = criterion(output[0], train_targets) * train_con_idx
            cls_loss = per_sample_loss.sum() / train_con_idx.sum().clamp_min(1.0)
            aff_loss = output[1]
            weight_loss = classifier_modulation_loss(model)
            loss = cls_loss * args.w1 + aff_loss * args.w2 + weight_loss * args.w3
            loss.backward()
            optimizer.step()

            confident_num += int(con_idx.sum().item())
            train_loss1 += cls_loss.item()
            train_loss2 += aff_loss.item()
            train_loss3 += weight_loss.item()
            batch_count += 1

        scheduler.step()
        print('[Target Epoch %d] Confident_Num: %d Classification Loss: %.3f '
              'Affinity Loss: %.3f Weight Loss: %.3f LR: %.6f' %
              (i, confident_num,
               train_loss1 / max(batch_count, 1),
               train_loss2 / max(batch_count, 1),
               train_loss3 / max(batch_count, 1),
               optimizer.param_groups[0]['lr']))

        val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), i, 'Validation'
        )
        if val_acc > best_target_val_acc:
            best_target_val_acc = val_acc
            save_checkpoint(
                target_best_path, model, optimizer, scheduler, i,
                best_target_val_acc, args
            )
            print('best target-stage validation accuracy %.4f' % best_target_val_acc)

    # Paper protocol: monitor/select using validation data, then report the
    # target TEST split exactly once using the best adapted checkpoint.
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
