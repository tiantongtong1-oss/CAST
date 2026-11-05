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
from raw_medoid_quantile import RawMedoidQuantileBank
from global_threshold import lambda_for_epoch, GlobalConfidenceMean, smooth_global_threshold
import image_utils as util
from randaugment import RandAugmentMC
# 总训练入口与调度器。串起数据、Source 预训练、EMA Teacher、
# 双视图伪标签、自适应阈值、KNN Reliability Rescue、Prototype Consistency、验证和测试

#随机种子
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

# 定义训练全部超参数
def parse_args(argv=None):
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
    parser.add_argument('--threshold_lambda', type=float, default=None,
                        help='optional fixed lambda; overrides dynamic schedule')
    parser.add_argument('--lambda_start', type=float, default=0.9)
    parser.add_argument('--lambda_end', type=float, default=1.0)
    parser.add_argument('--lambda_ramp_epochs', type=int, default=None,
                        help='default: total target epochs; ramp is zero-based')

    # Prototype-consistency regularizer retained from v1/v2.
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

    parser.add_argument('--rescue_warmup_epochs', type=int, default=3)
    args = parser.parse_args(argv)
    if args.lambda_ramp_epochs is None:
        args.lambda_ramp_epochs = max(args.epochs, 1)
    try:
        lambda_for_epoch(0, args.lambda_ramp_epochs, args.lambda_start,
                         args.lambda_end, args.threshold_lambda)
    except ValueError as error:
        parser.error(str(error))
    if args.rescue_warmup_epochs < 0:
        parser.error('rescue_warmup_epochs must be nonnegative')
    if args.pre_epochs < 0 or args.epochs < 0:
        parser.error('epoch counts must be nonnegative')
    if args.pre_epochs == 0 and not args.checkpoint:
        parser.error('--pre_epochs 0 requires --checkpoint with pretrained model weights')
    return args


#图像预处理、数据增强生成 弱增强 weak、强增强 strong、测试预处理 test
def build_transforms():
    # 对图像的 RGB 三个通道做标准化
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # 弱增强
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
        #强数据增强（随机选两种增强方式，强度为10 ）
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

# 分类器调制损失，对分类器各类别权重计算两两 余弦相似度，约束分类器权重关系
def classifier_modulation_loss(model):
    weight = model.fc.weight
    norm = torch.norm(weight, dim=1, keepdim=True).clamp_min(1e-12)
    cosine = weight.mm(weight.t()) / norm.mm(norm.t())
    matrix = cosine - torch.eye(weight.size(0), device=weight.device)
    return torch.mean((matrix + 1.0) / 2.0)


def calculate_target_thresholds(model, loader, class_num, threshold_lambda):
    """One shared tau from ALL target train samples, no agreement filtering."""
    stats = GlobalConfidenceMean()
    model.eval()
    with torch.no_grad():
        for original, augmented, _ in loader:
            logits1, _ = model(original.cuda(non_blocking=True), None, None,
                               mode='test', task='target')
            logits2, _ = model(augmented.cuda(non_blocking=True), None, None,
                               mode='test', task='target')
            conf1 = F.softmax(logits1, dim=1).max(dim=1).values
            conf2 = F.softmax(logits2, dim=1).max(dim=1).values
            scores = torch.minimum(conf1, conf2)
            if not torch.isfinite(scores).all():
                raise RuntimeError('Nonfinite target confidence during threshold scan')
            stats.add_batch(scores.double().sum().item(), scores.numel())
    threshold, mean = stats.threshold(threshold_lambda)
    # The existing selector indexes by class, so broadcast the SAME scalar.
    return torch.full((class_num,), threshold), mean, stats.count


def evaluate(model, loader, criterion, num_samples, epoch, split_name):
    # 在 FER val/test 上算 CE、accuracy 和 confusion matrix
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
                    teacher=None, prototype_bank=None, reliability_bank=None):
    # 保存 Student、optimizer、scheduler、epoch 等；
    # 目标阶段还可保存 EMA Teacher 和 PrototypeBank
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
    if reliability_bank is not None:
        state['reliability_bank'] = reliability_bank.state_dict()
    torch.save(state, path)


def initialize_source_prototypes(teacher, feature_hook, loader, prototype_bank):

    # 用 Teacher 在确定性的 RAF-DB source view 上抽特征，创建固定 源域原型

    teacher.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            teacher(imgs, None, None, mode='test', task='source')
            prototype_bank.accumulate_source(feature_hook.output, targets)
    prototype_bank.finalize_source()


@torch.no_grad()
def refresh_source_rescue(teacher, feature_hook, loader, radius_bank, prototype_bank):
    """Rebuild all source references in the current teacher feature space."""
    teacher.eval()
    device = next(teacher.parameters()).device
    features, labels = [], []
    for imgs, targets in loader:
        teacher(imgs.to(device), None, None, mode='test', task='source')
        features.append(feature_hook.output.detach().cpu())
        labels.append(targets.cpu())
    if not features:
        raise RuntimeError('Source loader is empty')
    radius_bank.set_source_memory(torch.cat(features), torch.cat(labels))
    prototype_bank.source_prototypes.copy_(radius_bank.prototypes)
    prototype_bank.source_initialized.copy_(radius_bank.initialized)
    prototype_bank.source_counts.copy_(radius_bank.counts)


def run_training():
    args = parse_args()
    model_path = os.path.join('./models', args.data1 + '_' + args.data2)
    os.makedirs(model_path, exist_ok=True)

    source_best_path = os.path.join(
        model_path,
        args.backbone + '_' + args.data1 + '_' + args.data2
        + '_globaltau_dynlambda_normmedoid_q90_r_or_source_best.pth'
    )
    target_best_path = os.path.join(
        model_path,
        args.backbone + '_' + args.data1 + '_' + args.data2
        + '_globaltau_dynlambda_normmedoid_q90_r_or_target_best.pth'
    )

    print('---------------------------------------------------------------------------------------')
    print('EMA + Original/Augmented + Global Min Mean + Dynamic Lambda + L2-Normalized Medoid Q90: '
          '%s with source %s and target %s' %
          (args.backbone, args.data1, args.data2))
    print('alpha:%s beta:%s gamma:%s ema:%s fixed_lambda:%s dynamic_lambda:[%s -> %s] ramp_epochs:%s' %
          (args.w1, args.w2, args.w3, args.ema_decay, args.threshold_lambda,
           args.lambda_start, args.lambda_end, args.lambda_ramp_epochs))
    print('prototype weight:%s temp:%s momentum:%s source_anchor:%s warmup:%s ramp:%s' %
          (args.proto_weight, args.proto_temperature, args.proto_momentum,
           args.proto_source_anchor, args.proto_warmup_epochs, args.proto_ramp_epochs))
    print('OR rescue: L2-normalized source/target radius gate; quantile=0.90 scale=1.0; refresh each epoch; raw source KNN k=11 strict majority')
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
    source_prototype = RafDataSet(
        args.source_path, phase='train', transform=test_transform,
        strong_transform=None, basic_aug=False
    )

    target_train = FER(
        args.target_path, phase='train', transform=test_transform,
        weak2_transform=weak_transform, strong_transform=strong_transform,
        basic_aug=False, return_index=True
    )
    target_threshold = FER(
        args.target_path, phase='train', transform=test_transform,
        weak2_transform=weak_transform, strong_transform=None, basic_aug=False
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
    # 输出分类 logits 和 source affinity loss
    model = Networks.Model(backbone=args.backbone, num_classes=class_num).cuda()
    # 训练组件初始化
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # 交叉熵损失
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
            # 清空梯度
            optimizer.zero_grad()
            output = model(imgs, targets, None, 'train', 'source')
            cls_loss = torch.mean(criterion(output[0], targets))
            aff_loss = output[1]
            weight_loss = classifier_modulation_loss(model)
            loss = cls_loss * args.w1 + aff_loss * args.w2 + weight_loss * args.w3
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            predicts = torch.argmax(output[0], dim=1)
            bingo_cnt += torch.eq(predicts, targets).sum().item()
            sample_count += targets.numel()
            train_loss1 += cls_loss.item()
            train_loss2 += aff_loss.item()
            train_loss3 += weight_loss.item()
            batch_count += 1
        # 更新学习率
        scheduler.step()
        train_acc = float(bingo_cnt) / max(sample_count, 1)
        print('[Source Epoch %d] Training accuracy: %.4f. Classification Loss: %.3f '
              'Affinity Loss: %.3f Weight Loss: %.3f LR: %.6f' %
              (i, train_acc,
               train_loss1 / max(batch_count, 1),
               train_loss2 / max(batch_count, 1),
               train_loss3 / max(batch_count, 1),
               optimizer.param_groups[0]['lr']))
        # 保存最佳的model
        val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), i, 'Validation'
        )
        if val_acc > best_source_val_acc:
            best_source_val_acc = val_acc
            save_checkpoint(source_best_path, model, optimizer, scheduler, i,
                            best_source_val_acc, args)
            print('best source-stage validation accuracy %.4f' % best_source_val_acc)
    # --checkpoint is weights initialization, not a full target-stage resume.
    if args.pre_epochs > 0:
        checkpoint = torch.load(source_best_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        # Weights were loaded above; start target training with a fresh optimizer.
        source_best_path = args.checkpoint
        best_source_val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), -1, 'Validation'
        )

    # 构建 EMA 教师模型
    teacher = create_ema_teacher(model).cuda()

    # 一个特征钩子，挂到 teacher.feature 和 model.feature 上
    # 作用：在前向传播时自动截获骨干网络的输出特征 fea，不需要改模型代码。
    teacher_feature_hook = FeatureHook(teacher.feature)
    student_feature_hook = FeatureHook(model.feature)

    # 初始化原型库（存储每个类别的特征原型）
    prototype_bank = PrototypeBank(
        class_num,
        model.fc.in_features,
        momentum=args.proto_momentum,
        source_anchor=args.proto_source_anchor,  #锚点权重
    ).cuda()

    # 源域数据初始化原型
    initialize_source_prototypes(
        teacher, teacher_feature_hook, prototype_loader_source, prototype_bank
    )
    print('source prototype counts: %s' %
          np.array2string(prototype_bank.source_counts.cpu().numpy(), separator=', '))

    reliability_bank = RawMedoidQuantileBank(
        prototype_bank.source_prototypes, prototype_bank.source_initialized
    ).cuda()
    best_target_val_acc = -1.0
    global_step = 0
    previous_global_threshold = None

    for i in range(args.epochs):
        current_lambda = lambda_for_epoch(
            i, args.lambda_ramp_epochs, args.lambda_start,
            args.lambda_end, args.threshold_lambda,
        )
        thresholds, global_mean, threshold_sample_count = calculate_target_thresholds(
            teacher, threshold_loader_target, class_num, current_lambda,
        )
        raw_global_threshold = current_lambda * global_mean
        previous_global_threshold = smooth_global_threshold(
            raw_global_threshold, previous_global_threshold, momentum=0.8,
        )
        thresholds.fill_(previous_global_threshold)
        current_proto_weight = prototype_weight_for_epoch(
            i, args.proto_weight, args.proto_warmup_epochs, args.proto_ramp_epochs
        )
        refresh_source_rescue(
            teacher, teacher_feature_hook, prototype_loader_source,
            reliability_bank, prototype_bank,
        )
        print('[Target Epoch %d] Global_R: %.8f Cutoff_R: %.8f Medoid_Memory_Row: %d Ready: %s Source_Memory: %d' %
              (i, reliability_bank.global_radius.item(), reliability_bank.global_radius.item(),
               reliability_bank.medoid_memory_index.item(), reliability_bank.global_ready.item(),
               reliability_bank.memory_labels.numel()))
        enable_knn_rescue = i >= args.rescue_warmup_epochs

        print('[Target Epoch %d] Global_Min_Confidence_Mean: %.6f Lambda: %.6f '
              'Raw_Global_Threshold: %.6f EMA_Momentum: 0.8 '
              'Global_Threshold: %.6f Threshold_Sample_Count: %d' %
              (i, global_mean, current_lambda, raw_global_threshold,
               thresholds[0].item(), threshold_sample_count))
        print('[Target Epoch %d] Radius/KNN OR rescue enabled: %s' %
              (i, enable_knn_rescue))

        model.train()
        source_train_iter = iter(train_loader_source)
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        train_proto_loss = 0.0
        batch_count = 0
        agreement_num = 0
        confidence_accept_num = 0
        radius_rescue_num = 0
        neighbor_rescue_num = 0
        knn_rescue_num = 0
        final_accept_num = 0
        knn_checked_num = 0
        prototype_agree_num = 0
        prototype_checked_num = 0
        prototype_similarity_sum = 0.0
        pseudo_distribution_confidence = np.zeros(class_num, dtype=np.int64)
        pseudo_distribution_rescued = np.zeros(class_num, dtype=np.int64)
        pseudo_distribution_final = np.zeros(class_num, dtype=np.int64)

        for original, augmented, strong, _, sample_ids in train_loader_target:
            try:
                source_imgs, source_targets = next(source_train_iter)
            except StopIteration:
                source_train_iter = iter(train_loader_source)
                source_imgs, source_targets = next(source_train_iter)

            original = original.cuda(non_blocking=True)
            augmented = augmented.cuda(non_blocking=True)
            strong = strong.cuda(non_blocking=True)

            teacher.eval()
            # Teacher predicts deterministic original + weak augmentation; original raw features serve
            # kNN可靠性判断和prototype更新使用
            with torch.no_grad():
                logits1, _ = teacher(original, None, None, 'test', 'target')
                teacher_features1 = teacher_feature_hook.output.detach()
                logits2, _ = teacher(augmented, None, None, 'test', 'target')
                teacher_features2 = teacher_feature_hook.output.detach()
                pseudo_targets, confidence_idx, agree_count = select_dual_view_pseudo_labels(
                    logits1, logits2, thresholds
                )
                # Keep original-view raw features here; the radius gate normalizes its own copy.
                teacher_reference_features = teacher_features1

                # 传统的高置信度伪标签
                confidence_mask = confidence_idx.bool()


                agreement_mask = torch.argmax(logits1, dim=1).eq(
                    torch.argmax(logits2, dim=1)
                )
                final_mask, radius_rescue_mask, neighbor_rescue_mask, neighbor_checked = (
                    reliability_bank.select(
                        teacher_reference_features, pseudo_targets, confidence_mask,
                        agreement_mask, enable_knn_rescue,
                    )
                )
                rescue_mask = radius_rescue_mask | neighbor_rescue_mask
                # Networks.forward selects idx == 1: retain a binary mask here.
                con_idx = final_mask.float()

                # 评估函数，用来统计在“可靠样本”上，模型特征与原型之间的预测一致性和相似度，会只检查 con_idx=1 的最终可靠目标样本
                proto_agree, proto_checked, proto_similarity = (
                    prototype_bank.agreement_stats(
                        teacher_reference_features, pseudo_targets, con_idx
                    )
                )
            # 统计当前epoch中不同阶段一共接纳了多少目标样本
            agreement_num += agree_count
            confidence_accept_num += int(confidence_mask.sum().item())
            radius_rescue_num += int(radius_rescue_mask.sum().item())
            neighbor_rescue_num += int(neighbor_rescue_mask.sum().item())
            knn_checked_num += int(neighbor_checked.sum().item())
            knn_rescue_num += int(rescue_mask.sum().item())
            final_accept_num += int(final_mask.sum().item())

            # 累计prototype诊断指标
            prototype_agree_num += proto_agree
            prototype_checked_num += proto_checked
            prototype_similarity_sum += proto_similarity * proto_checked

            # 统计伪标签在不同掩码筛选下的类别分布
            confidence_labels = pseudo_targets[confidence_mask].cpu().numpy()
            rescued_labels = pseudo_targets[rescue_mask].cpu().numpy()
            final_labels = pseudo_targets[final_mask].cpu().numpy()
            if confidence_labels.size > 0:
                pseudo_distribution_confidence += np.bincount(
                    confidence_labels, minlength=class_num
                ).astype(np.int64)
            if rescued_labels.size > 0:
                pseudo_distribution_rescued += np.bincount(
                    rescued_labels, minlength=class_num
                ).astype(np.int64)
            if final_labels.size > 0:
                pseudo_distribution_final += np.bincount(
                    final_labels, minlength=class_num
                ).astype(np.int64)

            model.train()
            # 给所有源域样本一个权重1，默认全可信
            source_con_idx = torch.ones(source_imgs.shape[0])
            # 源域图像+目标域strong augmentation
            train_imgs = torch.cat((source_imgs, strong.cpu()), dim=0).cuda(non_blocking=True)
            train_targets = torch.cat((source_targets, pseudo_targets.cpu()), dim=0).cuda(non_blocking=True)
            train_con_idx = torch.cat((source_con_idx, con_idx.cpu()), dim=0).cuda(non_blocking=True)

            # 清空模型参数的梯度
            optimizer.zero_grad()
            output = model(
                train_imgs, train_targets, train_con_idx, 'train', 'target',
                source_count=source_imgs.shape[0],
            )
            # 分类损失
            per_sample_loss = criterion(output[0], train_targets) * train_con_idx
            # 对真正有效的源域和目标域样本求平均分类损失
            cls_loss = per_sample_loss.sum() / train_con_idx.sum().clamp_min(1.0)
            aff_loss = output[1]
            # 分类器权重之间的正则项
            weight_loss = classifier_modulation_loss(model)
            # 把前面的源域特征切掉，只保留zt（student）
            target_student_features = student_feature_hook.output[source_imgs.shape[0]:]
            #prototype consistency原型一致性，让目标样本的 Student 特征靠近对应伪标签类别的 prototype
            proto_loss = prototype_bank.consistency_loss(
                target_student_features,
                pseudo_targets,
                con_idx,
                temperature=args.proto_temperature,
            )
            # 总损失
            loss = (
                cls_loss * args.w1
                + aff_loss * args.w2
                + weight_loss * args.w3
                + proto_loss * current_proto_weight
            )
            #反向传播
            loss.backward()
            optimizer.step()

            global_step += 1
            # 更新EMA Teacher
            update_ema_teacher(teacher, model, args.ema_decay, global_step)
            # 这是更新目标域prototype
            prototype_bank.update_target(
                teacher_reference_features, pseudo_targets, con_idx
            )

            train_loss1 += cls_loss.item()
            train_loss2 += aff_loss.item()
            train_loss3 += weight_loss.item()
            train_proto_loss += proto_loss.item()
            batch_count += 1
        # 更新学习率
        scheduler.step()
        # 求均值
        proto_agreement_rate = (
            float(prototype_agree_num) / float(prototype_checked_num)
            if prototype_checked_num > 0 else 0.0
        )
        proto_mean_similarity = (
            prototype_similarity_sum / float(prototype_checked_num)
            if prototype_checked_num > 0 else 0.0
        )
        print('[Target Epoch %d] Agreement_Num: %d Confidence_Accept_Num: %d '
              'Radius_Rescue_Num: %d Neighbor_Rescue_Num: %d Rescue_Num: %d '
              'Neighbor_Checked_Num: %d Final_Accept_Num: %d' %
              (i, agreement_num, confidence_accept_num, radius_rescue_num,
               neighbor_rescue_num, knn_rescue_num, knn_checked_num, final_accept_num))
        print('[Target Epoch %d] Pseudo_Distribution_Confidence: %s' %
              (i, np.array2string(pseudo_distribution_confidence, separator=', ')))
        print('[Target Epoch %d] Pseudo_Distribution_Rescued: %s' %
              (i, np.array2string(pseudo_distribution_rescued, separator=', ')))
        print('[Target Epoch %d] Pseudo_Distribution_Final: %s' %
              (i, np.array2string(pseudo_distribution_final, separator=', ')))
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
        # 在每个epoch 后用目标域 validation set做验证
        val_acc = evaluate(
            model, val_loader_target, criterion, len(target_val), i, 'Validation'
        )
        if val_acc > best_target_val_acc:
            best_target_val_acc = val_acc
            save_checkpoint(
                target_best_path, model, optimizer, scheduler, i,
                best_target_val_acc, args, teacher=teacher,
                prototype_bank=prototype_bank,
                reliability_bank=reliability_bank,
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
    from training_log import training_log
    with training_log():
        run_training()







