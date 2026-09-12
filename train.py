import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
from torchvision import transforms

import Networks
from dataset import RafDataSet, FER
from randaugment import RandAugmentMC
from training_utils import (classifier_weight_loss, classification_losses,
                            seed_everything, seed_worker, select_pseudo_labels,
                            target_affinity_weight, update_ema)


CLASS_NAMES = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data1', choices=['rafdb'], default='rafdb')
    parser.add_argument('--data2', choices=['fer'], default='fer')
    parser.add_argument('--idx', type=int, default=3, help='Legacy experiment identifier (not a split selector).')
    parser.add_argument('-c', '--checkpoint', help='Initialize model weights, then train on source.')
    parser.add_argument('--source_checkpoint', help='Skip source training and adapt these source weights directly.')
    parser.add_argument('--backbone', choices=['resnet18', 'resnet50', 'mobilenet_v2'], default='resnet18')
    parser.add_argument('--lr', type=float, default=0.001, help='Source Adam learning rate.')
    parser.add_argument('--target_lr', type=float, default=0.0003, help='Independent target Adam learning rate.')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30, help='Target adaptation epochs.')
    parser.add_argument('--source_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--w1', type=float, default=4)
    parser.add_argument('--w2', type=float, default=0.3, help='Source affinity weight.')
    parser.add_argument('--w3', type=float, default=0.1)
    parser.add_argument('--phi', type=float, default=1.4, help='CATM class threshold multiplier.')
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--target_w2', type=float, default=0.0, help='Target affinity maximum; zero disables it completely.')
    parser.add_argument('--affinity_warmup', type=int, default=5)
    parser.add_argument('--affinity_ramp', type=int, default=5)
    parser.add_argument('--target_cls_weight', type=float, default=0.5, help='Weight of normalized target CE relative to source CE.')
    parser.add_argument('--pseudo_ramp', type=int, default=5)
    parser.add_argument('--threshold_min', type=float, default=0.8)
    parser.add_argument('--threshold_max', type=float, default=0.95)
    parser.add_argument('--teacher_views', choices=['weak', 'legacy'], default='weak',
                        help='weak: resize/flip; legacy: source rotation/crop/erasing transforms.')
    parser.add_argument('--target_loss_reduction', choices=['normalized', 'legacy'], default='normalized',
                        help='legacy keeps the old reduction over all source/target samples for ablation.')
    parser.add_argument('--source_root', default='/workspace/ttt/code/test-upload-clean/datesets/raf-basic')
    parser.add_argument('--target_root', default='/workspace/ttt/code/data/fer2013')
    parser.add_argument('--output_dir', default='./models/rafdb_fer')
    parser.add_argument('--run_name', help='New subdirectory under output_dir; existing runs are never overwritten.')
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_pretrained', action='store_true', help='Do not download ImageNet initialization.')
    args = parser.parse_args(argv)
    if args.checkpoint and args.source_checkpoint:
        parser.error('Use only one of --checkpoint and --source_checkpoint.')
    if not 0 <= args.threshold_min <= args.threshold_max < 1:
        parser.error('Require 0 <= threshold_min <= threshold_max < 1.')
    if not 0 <= args.ema_decay < 1:
        parser.error('ema_decay must be in [0, 1).')
    if args.epochs < 0 or args.source_epochs < 0 or (args.source_epochs == 0 and not args.source_checkpoint):
        parser.error('Epochs must be nonnegative; source_epochs=0 requires --source_checkpoint.')
    if args.batch_size < 2 or args.workers < 0:
        parser.error('batch_size must be >= 2 and workers must be nonnegative.')
    if min(args.w1, args.w2, args.w3, args.target_w2, args.target_cls_weight,
           args.affinity_warmup, args.affinity_ramp, args.pseudo_ramp) < 0:
        parser.error('Loss weights and ramp lengths must be nonnegative.')
    if args.lr <= 0 or args.target_lr <= 0 or args.phi <= 0:
        parser.error('Learning rates and phi must be positive.')
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in {'.', '..'}):
        parser.error('run_name must be a directory name, not a path.')
    return args


def build_transforms():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return {
        'train': transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(20),
                                    transforms.RandomCrop(224, padding=32)], p=0.5),
            transforms.ToTensor(), normalize, transforms.RandomErasing(scale=(0.02, 0.25))]),
        'weak': transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]),
        'test': transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(), normalize]),
        'augment': transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(20),
                                    transforms.RandomCrop(224, padding=32)], p=0.5),
            RandAugmentMC(n=2, m=10), transforms.ToTensor(), normalize,
            transforms.RandomErasing(scale=(0.02, 0.25))]),
    }


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_manifest(dataset, root):
    # Identifies file membership/order and labels, not the image byte contents.
    records = [(os.path.relpath(p, root), int(y)) for p, y in zip(dataset.file_paths, dataset.label)]
    return {'count': len(records), 'class_counts': [int(x) for x in dataset.label_dis],
            'paths_labels_sha256': hashlib.sha256(json.dumps(records).encode()).hexdigest()}


def make_loader(dataset, args, shuffle, seed):
    if len(dataset) == 0:
        raise ValueError('Dataset is empty; check --source_root/--target_root and *.jpg files.')
    if shuffle and len(dataset) < args.batch_size:
        raise ValueError('Training dataset is smaller than batch_size; reduce --batch_size.')
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=shuffle, drop_last=shuffle, pin_memory=str(args.device).startswith('cuda'),
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate_only(model, loader, device):
    model.eval()
    confusion = torch.zeros((7, 7), dtype=torch.long)
    loss_sum, total = 0.0, 0
    for imgs, targets in loader:
        targets = targets.to(device)
        logits, _ = model(imgs.to(device), None, None, mode='test')
        loss_sum += torch.nn.functional.cross_entropy(logits, targets, reduction='sum').item()
        predictions = logits.argmax(1)
        confusion += torch.bincount((targets * 7 + predictions).cpu(), minlength=49).reshape(7, 7)
        total += targets.numel()
    if not total:
        raise ValueError('Cannot evaluate an empty dataset.')
    counts = confusion.sum(1)
    recalls = confusion.diag().float() / counts.clamp_min(1)
    return {'accuracy': confusion.diag().sum().item() / total, 'loss': loss_sum / total,
            'class_recall': recalls.tolist(), 'class_counts': counts.tolist(),
            'confusion_matrix': confusion.tolist(), 'num_samples': total}


def save_checkpoint(path, model, optimizer, scheduler, args, epoch, metrics,
                    model_kind, teacher=None, student=None):
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'args': vars(args), 'epoch': epoch,
               'metrics': metrics, 'model_kind': model_kind}
    if teacher is not None:
        payload['teacher'] = teacher.state_dict()
    if student is not None:
        payload['student'] = student.state_dict()
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def record_metrics(run_dir, row):
    with (run_dir / 'history.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')


def run_training(args=None):
    args = parse_args() if args is None else args
    seed_everything(args.seed)
    run_name = args.run_name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    tx = build_transforms()
    source_train = RafDataSet(args.source_root, 'train', transform=tx['train'],
                             strong_transform=tx['augment'], basic_aug=False)
    target_train = FER(args.target_root, 'train',
                       transform=tx['weak' if args.teacher_views == 'weak' else 'train'],
                       strong_transform=tx['augment'], basic_aug=False)
    target_test = FER(args.target_root, 'test', transform=tx['test'])
    source_loader = make_loader(source_train, args, True, args.seed)
    target_loader = make_loader(target_train, args, True, args.seed + 1)
    val_loader = make_loader(target_test, args, False, args.seed + 2)
    initial_path = args.source_checkpoint or args.checkpoint
    metadata = {'args': vars(args), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'evaluation_split': 'FER test (legacy target-validation protocol)',
                'source_train': dataset_manifest(source_train, args.source_root),
                'target_train': dataset_manifest(target_train, args.target_root),
                'target_test': dataset_manifest(target_test, args.target_root)}
    try:
        metadata['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip()
        metadata['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=Path(__file__).parent, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        metadata['git_commit'] = 'unavailable'
    if initial_path:
        metadata['initial_checkpoint_sha256'] = file_digest(initial_path)
    (run_dir / 'config.json').write_text(json.dumps(metadata, indent=2))
    print('Run directory:', run_dir, flush=True)
    print('Configuration:', json.dumps(vars(args), sort_keys=True), flush=True)
    print('Class order:', CLASS_NAMES)
    print('Evaluation: FER test, matching the previous target-validation protocol.')

    model = Networks.Model(backbone=args.backbone, num_classes=7,
                           pretrained=not (args.no_pretrained or initial_path)).to(device)
    if initial_path:
        checkpoint = torch.load(initial_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)
        print('Loaded weights:', initial_path)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    source_best_path = run_dir / 'source_best.pth'
    source_best = -1.0
    if not args.source_checkpoint:
        for epoch in range(args.source_epochs):
            model.train()
            sums = np.zeros(3)
            correct, total = 0, 0
            for imgs, _, targets in source_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                logits, affinity = model(imgs, targets, None, 'train', 'source',
                                         compute_affinity=args.w2 > 0)
                ce = torch.nn.functional.cross_entropy(logits, targets)
                modulation = classifier_weight_loss(model.fc.weight)
                loss = args.w1 * ce + args.w2 * affinity + args.w3 * modulation
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite source loss; training stopped before optimizer step.')
                loss.backward()
                optimizer.step()
                sums += [ce.item(), affinity.item(), modulation.item()]
                correct += logits.argmax(1).eq(targets).sum().item()
                total += targets.numel()
            metrics = evaluate_only(model, val_loader, device)
            print('[Source %d] train_acc=%.4f CE=%.3f affinity=%.3f val_acc=%.4f loss=%.3f LR=%.6f' %
                  (epoch, correct / total, sums[0] / len(source_loader), sums[1] / len(source_loader),
                   metrics['accuracy'], metrics['loss'], optimizer.param_groups[0]['lr']), flush=True)
            scheduler.step()
            record_metrics(run_dir, {'stage': 'source', 'epoch': epoch, **metrics})
            if metrics['accuracy'] > source_best:
                source_best = metrics['accuracy']
                save_checkpoint(source_best_path, model, optimizer, scheduler, args, epoch, metrics, 'source')
        checkpoint = torch.load(source_best_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

    # Independent optimizer/scheduler: the source best epoch no longer silently sets target LR.
    optimizer = torch.optim.Adam(model.parameters(), args.target_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    baseline = evaluate_only(model, val_loader, device)
    if args.source_checkpoint:
        save_checkpoint(source_best_path, model, optimizer, scheduler, args, -1, baseline, 'source')
    print('Source initialization target accuracy: %.4f' % baseline['accuracy'], flush=True)
    # Keep the source baseline as a candidate, even if all adaptation epochs regress.
    best = {'student': baseline['accuracy'], 'ema': baseline['accuracy'], 'overall': baseline['accuracy']}
    for name in ['target_student_best.pth', 'target_ema_best.pth', 'best.pth']:
        save_checkpoint(run_dir / name, model, optimizer, scheduler, args, -1, baseline,
                        'source', teacher=teacher, student=model)
    record_metrics(run_dir, {'stage': 'initial', 'epoch': -1, **baseline})

    # Reset target-stage randomness so training/reusing identical source weights is comparable.
    seed_everything(args.seed + 100)
    source_loader.generator.manual_seed(args.seed + 100)
    target_loader.generator.manual_seed(args.seed + 101)
    val_loader.generator.manual_seed(args.seed + 102)
    source_iter = iter(source_loader)
    for epoch in range(args.epochs):
        current_w2 = target_affinity_weight(epoch, args.target_w2, args.affinity_warmup, args.affinity_ramp)
        pseudo_scale = args.target_cls_weight * min(1.0, (epoch + 1) / max(args.pseudo_ramp, 1))
        sums = np.zeros(4)
        pseudo_counts = torch.zeros(7, dtype=torch.long)
        pseudo_correct = torch.zeros(7, dtype=torch.long)
        agreements, seen, weight_sum, selected_total = 0, 0, 0.0, 0
        lr = optimizer.param_groups[0]['lr']
        for w1, w2, strong, gt in target_loader:
            try:
                source_imgs, _, source_targets = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_imgs, _, source_targets = next(source_iter)
            teacher.eval()
            with torch.no_grad():
                logits1, _ = teacher(w1.to(device), None, None, mode='test')
                logits2, _ = teacher(w2.to(device), None, None, mode='test')
                labels, mask, weights, thresholds, agreement = select_pseudo_labels(
                    logits1, logits2, epoch, args.epochs, args.phi,
                    args.threshold_min, args.threshold_max)
            # Ground-truth target labels are used ONLY for the following diagnostics.
            labels_cpu, mask_cpu = labels.cpu(), mask.cpu()
            pseudo_counts += torch.bincount(labels_cpu[mask_cpu], minlength=7)
            correct_mask = mask_cpu & labels_cpu.eq(gt)
            pseudo_correct += torch.bincount(labels_cpu[correct_mask], minlength=7)
            agreements += agreement.sum().item()
            seen += mask.numel()
            weight_sum += weights.sum().item()
            selected_total += mask.sum().item()

            model.train()
            source_targets = source_targets.to(device)
            n_source = source_targets.numel()
            images = torch.cat([source_imgs.to(device), strong.to(device)])
            targets = torch.cat([source_targets, labels])
            align_mask = torch.cat([torch.ones(n_source, device=device, dtype=torch.bool),
                                    mask & (weights >= 0.75)])
            optimizer.zero_grad()
            logits, affinity = model(images, targets, align_mask, 'train', 'target',
                                     source_count=n_source, compute_affinity=current_w2 > 0)
            source_ce, target_ce = classification_losses(logits, source_targets, labels, weights)
            if args.target_loss_reduction == 'legacy':
                ce_weights = torch.cat([torch.ones(n_source, device=device), weights])
                ce = (torch.nn.functional.cross_entropy(logits, targets, reduction='none') * ce_weights).mean()
            else:
                ce = 0.5 * (source_ce + pseudo_scale * target_ce)
            loss = args.w1 * ce + current_w2 * affinity + args.w3 * classifier_weight_loss(model.fc.weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite target loss; training stopped before optimizer step.')
            loss.backward()
            optimizer.step()
            update_ema(model, teacher, args.ema_decay)
            sums += [source_ce.item(), target_ce.item(), affinity.item(), ce.item()]
        scheduler.step()
        stats = {'pseudo_accuracy': pseudo_correct.sum().item() / max(selected_total, 1),
                 'pseudo_class_accuracy': (pseudo_correct.float() / pseudo_counts.clamp_min(1)).tolist(),
                 'pseudo_class_counts': pseudo_counts.tolist(), 'selected': selected_total,
                 'agreement': agreements / max(seen, 1), 'mean_pseudo_weight': weight_sum / max(selected_total, 1),
                 'source_ce': sums[0] / len(target_loader), 'target_ce': sums[1] / len(target_loader),
                 'affinity': sums[2] / len(target_loader), 'lr': lr,
                 'target_w2': current_w2, 'target_cls_scale': pseudo_scale}
        print('[Target %d] selected=%d pseudo_acc=%.4f agreement=%.4f source_CE=%.3f target_CE=%.3f w2=%.4f LR=%.6f' %
              (epoch, selected_total, stats['pseudo_accuracy'], stats['agreement'], stats['source_ce'],
               stats['target_ce'], current_w2, lr), flush=True)
        print('Pseudo class counts:', stats['pseudo_class_counts'])
        for kind, candidate in [('student', model), ('ema', teacher)]:
            metrics = evaluate_only(candidate, val_loader, device)
            print('[Target %d] %s accuracy=%.4f loss=%.3f recall=%s' %
                  (epoch, kind, metrics['accuracy'], metrics['loss'],
                   [round(x, 4) for x in metrics['class_recall']]), flush=True)
            record_metrics(run_dir, {'stage': 'target', 'epoch': epoch, 'model_kind': kind,
                                     **stats, **metrics})
            if metrics['accuracy'] > best[kind]:
                best[kind] = metrics['accuracy']
                save_checkpoint(run_dir / ('target_%s_best.pth' % kind), candidate, optimizer, scheduler,
                                args, epoch, metrics, kind, teacher=teacher, student=model)
            if metrics['accuracy'] > best['overall']:
                best['overall'] = metrics['accuracy']
                save_checkpoint(run_dir / 'best.pth', candidate, optimizer, scheduler,
                                args, epoch, metrics, kind, teacher=teacher, student=model)
    (run_dir / 'summary.json').write_text(json.dumps(best, indent=2))
    print('Best student: %.4f | Best EMA: %.4f | best_acc %.4f' %
          (best['student'], best['ema'], best['overall']), flush=True)
    print('Best checkpoint:', run_dir / 'best.pth', flush=True)
    return best


if __name__ == '__main__':
    run_training()
