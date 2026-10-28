"""Offline v27 acceptance quality on FER train; no training or role promotion."""
import argparse
import csv
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np

from competition_gate import competition_audit
from neighbor_vote import audit_vote, rescue_union
from pseudo_label_metrics import CLASS_NAMES


DEFAULTS = dict(backbone='mobilenet_v2', threshold_base=.85,
                threshold_std_scale=.5, threshold_smoothing=.8,
                threshold_min=.85, threshold_max=.99,
                source_radius_quantile=.9, rescue_probability_margin=.1,
                rescue_prototype_margin=.02, rescue_vote_margin=3)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source_path', default='/workspace/ttt/code/test-upload-clean/datesets/raf-basic')
    p.add_argument('--target_path', default='/workspace/ttt/code/data/fer2013')
    p.add_argument('--weights', choices=['auto', 'model', 'ema_teacher'], default='auto')
    p.add_argument('--backbone', choices=['mobilenet_v2', 'resnet18', 'resnet50'])
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=1314)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--rescue_mode', choices=['enabled', 'disabled'], default='enabled',
                   help='enabled evaluates post-warmup rescue; disabled evaluates confidence only')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--threshold', type=float, help='use a known fixed global threshold without scanning')
    group.add_argument('--previous_tau', type=float,
                       help='optional previous-epoch tau for the existing EMA scanner')
    for key in DEFAULTS:
        if key != 'backbone':
            p.add_argument('--' + key, type=int if key == 'rescue_vote_margin' else float,
                           default=None, help='override checkpoint args, otherwise use v27 default')
    a = p.parse_args(argv)
    if a.batch_size < 1 or a.workers < 0:
        p.error('batch_size must be positive and workers nonnegative')
    for key in ('threshold', 'previous_tau'):
        v = getattr(a, key)
        if v is not None and (not np.isfinite(v) or not 0 < v <= 1):
            p.error(key + ' must be finite and in (0,1]')
    return a


def settings_from_checkpoint(args, checkpoint):
    saved = checkpoint.get('args', {})
    if not isinstance(saved, dict):
        saved = vars(saved)
    values = {k: saved.get(k, v) for k, v in DEFAULTS.items()}
    values.update({k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None})
    a = SimpleNamespace(**values)
    if a.backbone not in ('mobilenet_v2', 'resnet18', 'resnet50'):
        raise ValueError('unsupported checkpoint backbone')
    if not all(np.isfinite(v) for k, v in values.items() if k != 'backbone'):
        raise ValueError('nonfinite checkpoint/CLI setting')
    if not 0 < a.threshold_min <= a.threshold_base <= a.threshold_max <= 1:
        raise ValueError('require 0 < threshold_min <= threshold_base <= threshold_max <= 1')
    if a.threshold_std_scale < 0 or not 0 <= a.threshold_smoothing < 1:
        raise ValueError('invalid threshold statistics settings')
    if not 0 < a.source_radius_quantile <= 1:
        raise ValueError('source_radius_quantile must be in (0,1]')
    if not (0 <= a.rescue_probability_margin <= 1 and 0 <= a.rescue_prototype_margin <= 2
            and int(a.rescue_vote_margin) == a.rescue_vote_margin and 0 <= a.rescue_vote_margin <= 11):
        raise ValueError('invalid competition margins')
    return a


def select_weight_key(checkpoint, requested):
    key = ('ema_teacher' if 'ema_teacher' in checkpoint else 'model') if requested == 'auto' else requested
    if key not in checkpoint:
        raise KeyError('checkpoint does not contain ' + key)
    return key


def rescue_selection(prob1, prob2, features, labels, ids, confidence, agreement,
                     region_ok, bank, config, means, ready, settings, enabled):
    """Same v27 post-warmup rescue gates. Does not accept target truth."""
    rescued = np.zeros(len(labels), dtype=bool)
    idx = np.flatnonzero(agreement & ~confidence)
    if enabled and len(idx):
        neighbors = bank.evaluate(features[idx], ids[idx], config)
        vote = audit_vote(neighbors, labels[idx], 6)
        competition = competition_audit(
            prob1[idx], prob2[idx], features[idx], labels[idx], means, ready,
            neighbors['votes'], neighbors['available'],
            settings.rescue_probability_margin, settings.rescue_prototype_margin,
            settings.rescue_vote_margin)
        rescued[idx] = rescue_union(region_ok[idx], vote['pass_mask']) & competition['pass_mask']
    return rescued


def summarize(predictions, truth, masks):
    """Per-predicted-class precision; empty selections have no accuracy."""
    predictions, truth = np.asarray(predictions), np.asarray(truth)
    rows = []
    for stage, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        n = np.bincount(predictions[mask], minlength=7)
        k = np.bincount(predictions[mask & (predictions == truth)], minlength=7)
        for c, name in list(enumerate(CLASS_NAMES)) + [(-1, 'ALL')]:
            count = int(n[c]) if c >= 0 else int(n.sum())
            correct = int(k[c]) if c >= 0 else int(k.sum())
            rows.append(dict(stage=stage, class_id=c, class_name=name,
                             accepted_count=count, correct_count=correct,
                             pseudo_accuracy=correct/count if count else None))
    return rows


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    # Keep --help and unit tests usable without the FER/PyTorch environment.
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import Networks
    from dataset import FER, RafDataSet
    from prototype_utils import FeatureHook
    from source_prototype_radius import SourcePrototypeRadiusBank
    from dual_domain_ds import DSConfig
    from ema_utils import select_dual_view_pseudo_labels
    from train import (build_transforms, calculate_global_mean_std_threshold,
                       refresh_source_radius_bank, refresh_ds_source_bank)

    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('choose an empty --output_dir to preserve previous results')
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass --device cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    settings = settings_from_checkpoint(args, checkpoint)
    weight_key = select_weight_key(checkpoint, args.weights)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = Networks.Model(backbone=settings.backbone, num_classes=7, pretrained=False).to(device)
    model.load_state_dict(checkpoint[weight_key], strict=True)
    model.eval()
    weak, _, test = build_transforms()
    source = RafDataSet(args.source_path, phase='train', transform=test, basic_aug=False)
    target = FER(args.target_path, phase='train', transform=weak, weak2_transform=weak,
                 basic_aug=False, return_index=True)
    # Reset after dataset constructors, which can alter global RNG state.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def loader(dataset):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers,
                          generator=torch.Generator().manual_seed(args.seed))

    hook = FeatureHook(model.feature)
    radius_bank = SourcePrototypeRadiusBank(7, model.fc.in_features,
                                           settings.source_radius_quantile).to(device)
    config = DSConfig(k=11, alpha=1.)  # v27 unweighted KNN uses k only.
    try:
        with torch.no_grad():
            print('Building source prototypes/radii and K=11 reference bank...', flush=True)
            refresh_source_radius_bank(model, hook, loader(source), radius_bank)
            bank = refresh_ds_source_bank(model, hook, loader(source), 7)
            scan = None
            if args.threshold is not None:
                tau = args.threshold
                threshold_mode = 'explicit_fixed'
            else:
                print('Scanning full FER train for global mean/std threshold...', flush=True)
                scan = calculate_global_mean_std_threshold(
                    model, loader(target), previous_tau=args.previous_tau,
                    std_scale=settings.threshold_std_scale, smoothing=settings.threshold_smoothing,
                    threshold_min=settings.threshold_min, threshold_max=settings.threshold_max,
                    fallback=settings.threshold_base)
                tau = scan[0]
                threshold_mode = 'recomputed_with_previous_tau' if args.previous_tau is not None else 'recomputed_without_history'
            thresholds = torch.full((7,), tau, dtype=torch.float32)
            print('Weights=%s threshold=%.6f rescue=%s' % (weight_key, tau, args.rescue_mode), flush=True)
            count = len(target)
            predictions = np.full(count, -1, dtype=np.int64)
            masks = {s: np.zeros(count, dtype=bool) for s in ('confidence', 'rescued', 'final')}
            seen = np.zeros(count, dtype=bool)
            samples = []
            for batch, (view1, view2, _ignored_truth, ids) in enumerate(loader(target)):
                logits1, _ = model(view1.to(device), None, None, 'test', 'target')
                f1 = hook.output.detach().clone()
                logits2, _ = model(view2.to(device), None, None, 'test', 'target')
                f2 = hook.output.detach()
                z = F.normalize(F.normalize(f1, dim=1) + F.normalize(f2, dim=1), dim=1)
                labels, accepted, _ = select_dual_view_pseudo_labels(logits1, logits2, thresholds)
                agreement = labels.eq(logits2.argmax(dim=1))
                region = radius_bank.in_distribution(z, labels)
                ids_np, labels_np = ids.numpy(), labels.cpu().numpy()
                direct = accepted.bool().cpu().numpy()
                rescued = rescue_selection(
                    logits1.softmax(1).cpu().numpy(), logits2.softmax(1).cpu().numpy(),
                    z.cpu().numpy(), labels_np, ids_np, direct, agreement.cpu().numpy(),
                    region.cpu().numpy(), bank, config, radius_bank.class_means.cpu().numpy(),
                    radius_bank.distribution_initialized.cpu().numpy(), settings,
                    args.rescue_mode == 'enabled')
                if seen[ids_np].any():
                    raise RuntimeError('duplicate target IDs')
                seen[ids_np] = True
                predictions[ids_np] = labels_np
                for stage, mask in zip(masks, (direct, rescued, direct | rescued)):
                    masks[stage][ids_np] = mask
                min_conf = torch.minimum(logits1.softmax(1).max(1).values,
                                         logits2.softmax(1).max(1).values).cpu().numpy()
                # Target truth is accessed only after all selection decisions.
                for j, sample_id in enumerate(ids_np):
                    truth = int(target.label[sample_id])
                    samples.append(dict(sample_id=int(sample_id), path=target.file_paths[sample_id],
                                        pseudo_label=int(labels_np[j]), true_label=truth,
                                        correct=bool(labels_np[j] == truth), min_confidence=float(min_conf[j]),
                                        threshold=tau, agreement=bool(agreement[j].item()),
                                        confidence=bool(direct[j]), rescued=bool(rescued[j]),
                                        final=bool(direct[j] | rescued[j])))
                if batch % 25 == 0:
                    print('Evaluated %d/%d target images' % (int(seen.sum()), count), flush=True)
            if not seen.all():
                raise RuntimeError('incomplete target coverage')
            rows = summarize(predictions, np.asarray(target.label), masks)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / 'pseudo_label_metrics.csv', rows)
        write_csv(out / 'samples.csv', samples)
        summary = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                       checkpoint_epoch=checkpoint.get('epoch'), weights=weight_key,
                       options=vars(args), effective_settings=vars(settings), threshold=tau,
                       threshold_mode=threshold_mode, threshold_scan=scan,
                       source_references=len(bank.x), target_samples=count,
                       metrics=rows, history_metrics_available=False,
                       note='Frozen-checkpoint snapshot on FER train. Rescue enabled means post-warmup rules. '
                            'No stable promotion, anchors or actual supervised-mask reconstruction. '
                            'Checkpoint roles are deliberately not advanced. Threshold history is not saved '
                            'by v27; supply --threshold or --previous_tau if known. '
                            'Target truth is used for reporting only. Fresh weak views can differ from training.')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        for row in rows:
            accuracy = row['pseudo_accuracy']
            print('%-10s %-9s accepted=%6d correct=%6d accuracy=%s' %
                  (row['stage'], row['class_name'], row['accepted_count'], row['correct_count'],
                   'N/A' if accuracy is None else '%.2f%%' % (100 * accuracy)))
        print('Saved:', out.resolve())
    finally:
        hook.close()


if __name__ == '__main__':
    run(parse_args())
