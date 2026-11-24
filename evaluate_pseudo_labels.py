"""Read-only pseudo-label audit for experiment/emotion-prototype-radius-q85-q90.

Run from repository root; see --help. No training/optimizer/checkpoint writes.
Target labels are read ONLY after every prediction and selection is frozen.
This is a snapshot audit, NOT recovery of historical online training masks:
training changes the EMA teacher within an epoch and uses stale epoch memory.
Do not tune hyperparameters/checkpoints on this diagnostic ground truth.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_overlap(source_paths, train_paths, heldout_paths):
    """Reject source/target overlap; count target train/val overlap without blocking."""
    seen = {}
    overlap_count = 0
    for group, paths in [('source_train', source_paths), ('target_train', train_paths),
                         ('target_heldout', heldout_paths)]:
        for path in paths:
            path = Path(path)
            matched = False
            for key in [('path', str(path.resolve())), ('sha256', digest(path))]:
                previous = seen.get(key)
                if previous is not None and previous != group:
                    if 'source_train' in (previous, group):
                        raise ValueError('Cross-domain image overlap: ' + previous + ' / ' + group)
                    matched = True
                if previous is None:
                    seen[key] = group
            overlap_count += int(matched)
    return overlap_count


def summarize(predictions, truth, masks):
    result = {}
    for name, mask in masks.items():
        rows = [i for i, keep in enumerate(mask) if keep]
        correct = sum(predictions[i] == truth[i] for i in rows)
        by_class = []
        for c in range(7):
            indices = [i for i in rows if predictions[i] == c]
            hits = sum(predictions[i] == truth[i] for i in indices)
            by_class.append(dict(pseudo_class=c, count=len(indices), correct=hits,
                                 accuracy=hits / len(indices) if indices else None))
        result[name] = dict(count=len(rows), correct=correct,
                            accuracy=correct / len(rows) if rows else None,
                            coverage=len(rows) / len(predictions), by_pseudo_class=by_class)
    return result



STAGE_DESCRIPTIONS = {
    'before': '全部目标训练样本',
    'agreement': '双视图预测一致',
    'confidence': '通过高置信度筛选（含双视图一致条件）',
    'confidence_rejected': '未通过高置信度筛选（含双视图不一致）',
    'low_conf_agreement': '双视图一致但未通过高置信度筛选',
    'radius_candidates': '低置信度一致样本中通过半径筛选（KNN前）',
    'radius_rejected': '低置信度一致样本中未通过半径筛选（半径启用时）',
    'neither_candidate': '高置信度与半径两条候选路径均未通过（KNN前）',
    'confidence_knn': '高置信度路径通过KNN',
    'confidence_knn_rejected': '高置信度候选未通过KNN',
    'radius_final': '半径救援路径通过严格双域KNN',
    'radius_knn_rejected': '半径候选未通过严格双域KNN',
    'final': '两条路径经过KNN后的最终接纳并集',
    'rejected': '最终未接纳：两条完整筛选路径均未通过',
}


def build_stage_masks(confidence, agree, radius, final, radius_enabled):
    """Label-free, disjoint path statistics; radius failures exclude ineligible rows."""
    zero = confidence & ~confidence
    low = agree & ~confidence
    return {
        'before': confidence | ~confidence,
        'agreement': agree,
        'confidence': confidence,
        'confidence_rejected': ~confidence,
        'low_conf_agreement': low,
        'radius_candidates': radius,
        'radius_rejected': low & ~radius if radius_enabled else zero,
        'neither_candidate': ~(confidence | radius),
        'confidence_knn': confidence & final,
        'confidence_knn_rejected': confidence & ~final,
        'radius_final': radius & final,
        'radius_knn_rejected': radius & ~final,
        'final': final,
        'rejected': ~final,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='Trusted local checkpoint')
    p.add_argument('--source_path', required=True)
    p.add_argument('--target_path', required=True)
    p.add_argument('--output_dir', required=True, help='New directory, outside dataset roots')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=1314)
    p.add_argument('--backbone', choices=['mobilenet_v2', 'resnet18', 'resnet50'])
    p.add_argument('--weights', choices=['auto', 'model', 'ema_teacher'], default='auto')
    p.add_argument('--epoch_index', type=int, default=None, help='Zero-based radius/schedule epoch; defaults to checkpoint epoch')
    threshold = p.add_mutually_exclusive_group(required=True)
    threshold.add_argument('--threshold', type=float, help='Actual smoothed Global_Threshold from training log')
    threshold.add_argument('--recompute_threshold', action='store_true', help='Use lambda * current two-view mean, WITHOUT historical threshold EMA')
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        p.error('batch_size must be positive and workers nonnegative')
    if args.threshold is not None and (not math.isfinite(args.threshold) or args.threshold < 0):
        p.error('threshold must be finite and nonnegative')
    out = Path(args.output_dir).resolve()
    for root in [args.source_path, args.target_path]:
        if out == Path(root).resolve() or Path(root).resolve() in out.parents:
            p.error('output_dir must be outside dataset roots')
    if out.exists():
        p.error('output_dir already exists')

    import cv2
    import numpy as np
    import torch
    import Networks
    from dataset import RafDataSet, FER
    from train import build_transforms
    from ema_utils import select_dual_view_pseudo_labels
    from fused_knn import FusedKNNBank
    from global_threshold import lambda_for_epoch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    # Repository checkpoints contain tensors, primitives, dicts and lists only.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint.get('args', {})
    epoch = args.epoch_index if args.epoch_index is not None else checkpoint.get('epoch')
    if not isinstance(epoch, int) or epoch < 0:
        p.error('Specify a nonnegative --epoch_index')
    backbone = args.backbone or config.get('backbone', 'mobilenet_v2')
    key = args.weights
    if key == 'auto':
        key = 'ema_teacher' if 'ema_teacher' in checkpoint else 'model'
    if key not in checkpoint:
        p.error('Checkpoint missing ' + key)
    weak, _, test = build_transforms()
    source = RafDataSet(args.source_path, phase='train', transform=test, basic_aug=False)
    root = Path(args.target_path)
    paths = sorted(root.glob('train/*/*.jpg'))
    if not paths:
        p.error('No target train/*/*.jpg images')
    np.random.RandomState(2000).shuffle(paths)
    # Validation only. Never enumerate or read the test split.
    validation_split = 'val'
    heldout = sorted(root.glob('val/*/*.jpg'))
    if not heldout:
        validation_split = 'validation'
        heldout = sorted(root.glob('validation/*/*.jpg'))
    if not heldout:
        p.error('No val/validation images found')
    overlap_count = audit_overlap(source.file_paths, paths, heldout)
    print(f'Validation duplicate images overlapping train: {overlap_count}; continuing without removing rows. Test split not accessed.')

    class UnlabeledViews(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            image = cv2.imread(str(paths[index]))
            if image is None:
                raise FileNotFoundError(str(paths[index]))
            image = image[:, :, ::-1]
            # No directory labels are parsed here or passed to the model/bank.
            return test(image), weak(image), index

    def loader(ds):
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.workers)

    model = Networks.Model(backbone=backbone, num_classes=7, pretrained=False).to(device)
    model.load_state_dict(checkpoint[key], strict=True)
    model.eval().requires_grad_(False)
    bank = FusedKNNBank(model.fc.in_features, 7,
                        config.get('knn_support_threshold', .7)).to(device)
    with torch.inference_mode():
        sf, sy = [], []
        for images, labels in loader(source):
            _, features = model(images.to(device), None, None, mode='test', task='source')
            sf.append(features.cpu())
            sy.append(labels)
        bank.set_source_memory(torch.cat(sf), torch.cat(sy))
        if not bank.class_center_ready.all():
            raise ValueError('Each source class requires at least two valid features and a valid prototype')
        if bank.source_labels.numel() < bank.k or len(paths) <= bank.k:
            raise ValueError('Insufficient neighbors for self-excluding kNN')
        logits_a, logits_b, features, all_ids = [], [], [], []
        for original, augmented, ids in loader(UnlabeledViews()):
            a, f = model(original.to(device), None, None, mode='test', task='target')
            b, _ = model(augmented.to(device), None, None, mode='test', task='target')
            if not (torch.isfinite(a).all() and torch.isfinite(b).all() and
                    torch.isfinite(f).all() and (f.norm(dim=1) > 1e-8).all()):
                raise ValueError('Invalid logits/features')
            logits_a.append(a.cpu())
            logits_b.append(b.cpu())
            features.append(f.cpu())
            all_ids.append(ids)
        a, b, f, ids = map(torch.cat, [logits_a, logits_b, features, all_ids])
        if not torch.equal(ids, torch.arange(len(paths))):
            raise ValueError('Sample order mismatch')
        bank.set_target_memory(f, a.softmax(1), ids)
        if bank.memory_ids.numel() != len(paths):
            raise ValueError('Target memory unexpectedly dropped rows')
        mean = torch.cat([a.softmax(1).max(1).values, b.softmax(1).max(1).values]).double().mean().item()
        tau = args.threshold
        if tau is None:
            lam = lambda_for_epoch(epoch, config.get('lambda_ramp_epochs') or max(config.get('epochs', 30), 1),
                                   config.get('lambda_start', .9), config.get('lambda_end', 1.),
                                   config.get('threshold_lambda'))
            tau = lam * mean
        masks = {name: [] for name in STAGE_DESCRIPTIONS}
        predictions = []
        radius_enabled = epoch >= config.get('rescue_warmup_epochs', 3)
        for start in range(0, len(paths), args.batch_size):
            sl = slice(start, start + args.batch_size)
            aa, bb, ff = a[sl].to(device), b[sl].to(device), f[sl].to(device)
            pred, confidence, _ = select_dual_view_pseudo_labels(aa, bb, torch.full((7,), tau))
            agree = aa.argmax(1).eq(bb.argmax(1))
            confidence = confidence.bool()
            radius = agree & ~confidence & bank.radius_accepts(ff, pred) if radius_enabled else torch.zeros_like(agree)
            final = bank.select(ff, pred, confidence, agree, ids[sl].to(device), radius_candidates=radius)[0]
            predictions.extend(pred.cpu().tolist())
            for name, mask in build_stage_masks(confidence, agree, radius, final, radius_enabled).items():
                masks[name].extend(mask.cpu().tolist())

    # Boundary: all predictions/masks are now fixed. Read target truth ONLY here.
    truth = [FER.FER_TO_CAST[int(path.parent.name)] for path in paths]
    report = summarize(predictions, truth, masks)
    for name, values in report.items():
        values['description'] = STAGE_DESCRIPTIONS[name]
    assert report['confidence_knn']['count'] + report['radius_final']['count'] == report['final']['count']
    assert report['confidence_knn']['correct'] + report['radius_final']['correct'] == report['final']['correct']
    metadata = dict(scope='fixed-checkpoint target TRAIN snapshot; not historical training accuracy',
                    selected_weights=key, epoch_index=epoch, threshold=tau,
                    metric_definitions=STAGE_DESCRIPTIONS,
                    radius_failure_scope='agreement & ~confidence; empty when radius disabled',
                    threshold_mode='log-supplied' if args.threshold is not None else 'recomputed WITHOUT historical EMA',
                    two_view_mean=mean, radius_enabled=radius_enabled,
                    checkpoint_sha256=digest(args.checkpoint), seed=args.seed,
                    support_threshold=bank.support_threshold, k=bank.k,
                    radius_groups={'positive': [0, 3, 6], 'negative': [1, 2, 4, 5]},
                    radius_center='L2-normalized mean of L2-normalized source features by class',
                    radius_rule='positive=balancedQ85; negative=balancedQ90',
                    group_radii=bank.group_radii.cpu().tolist(),
                    group_radius_ready=bank.group_radius_ready.cpu().tolist(),
                    rescue_rule='both source and target kNN argmax agree; shared support threshold',
                    cross_split_byte_and_path_overlap_check='validation-only; target duplicates allowed',
                    validation_split=validation_split, validation_overlap_images=overlap_count,
                    test_accessed=False, heldout_count=len(heldout),
                    labels='surprise,fear,disgust,happy,sad,angry,neutral',
                    limitation='Cannot certify upstream checkpoint provenance or detect transformed near-duplicates; no historical masks or EMA threshold state saved.')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(dict(metadata=metadata, metrics=report), indent=2), encoding='utf-8')
    import csv
    with (out / 'stages.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['stage', 'description', 'count', 'correct', 'accuracy', 'coverage'])
        for name, values in report.items():
            writer.writerow([name, values['description'], values['count'], values['correct'],
                             values['accuracy'], values['coverage']])
    # Aggregate-only output: no images, paths, per-sample predictions or truth exported.
    for name, values in report.items():
        acc = 'N/A' if values['accuracy'] is None else format(values['accuracy'], '.2%')
        print(f"{name} [{values['description']}]: {values['correct']}/{values['count']} accuracy={acc} coverage={values['coverage']:.2%}")
    print('Snapshot audit saved:', out / 'summary.json')


if __name__ == '__main__':
    main()
