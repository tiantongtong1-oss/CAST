"""Read-only checkpoint diagnostic for RAF -> FER source-only DS; no training."""
import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
from dual_domain_ds import DSConfig, EvidenceBank, audit_source
from ds_diagnostics import diagnose_query


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source_path', default='/workspace/ttt/code/test-upload-clean/datesets/raf-basic')
    p.add_argument('--target_path', default='/workspace/ttt/code/data/fer2013')
    p.add_argument('--backbone', default='mobilenet_v2')
    p.add_argument('--weights', choices=['auto', 'model', 'ema_teacher'], default='auto')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=1314)
    p.add_argument('--max_samples', type=int, default=500)
    p.add_argument('--output_dir', default='./ds_debug/source_only_checkpoint')
    p.add_argument('--ds_k', type=int, default=11)
    p.add_argument('--ds_min_neighbors', type=int, default=3)
    p.add_argument('--ds_max_distance', type=float, default=.5)
    p.add_argument('--ds_bandwidth', type=float, default=.25)
    p.add_argument('--ds_strength_cap', type=float, default=.95)
    p.add_argument('--ds_support_threshold', type=float, default=.6)
    p.add_argument('--ds_margin_threshold', type=float, default=.1)
    p.add_argument('--ds_uncertainty_threshold', type=float, default=.3)
    p.add_argument('--ds_conflict_threshold', type=float, default=.5)
    p.add_argument('--source_radius_quantile', type=float, default=.9)
    p.add_argument('--threshold_base', type=float, default=.85)
    p.add_argument('--threshold_beta', type=float, default=.5)
    p.add_argument('--threshold_margin', type=float, default=.02)
    p.add_argument('--threshold_min', type=float, default=.8)
    p.add_argument('--threshold_max', type=float, default=.95)
    args = p.parse_args(argv)
    if args.max_samples < 1 or args.batch_size < 1 or args.workers < 0:
        p.error('sample/batch counts must be positive; workers nonnegative')
    if not 0 < args.source_radius_quantile <= 1:
        p.error('source_radius_quantile must be in (0,1]')
    if not 0 < args.threshold_min <= args.threshold_base <= args.threshold_max <= 1:
        p.error('invalid threshold bounds')
    if not np.isfinite([args.threshold_beta, args.threshold_margin]).all() or min(args.threshold_beta, args.threshold_margin) < 0:
        p.error('threshold beta/margin must be finite and nonnegative')
    try:
        config_from_args(args)
    except ValueError as error:
        p.error(str(error))
    return args


def config_from_args(a):
    return DSConfig(k=a.ds_k, min_neighbors=a.ds_min_neighbors, max_distance=a.ds_max_distance,
                    bandwidth=a.ds_bandwidth, strength_cap=a.ds_strength_cap, alpha=1.,
                    support=a.ds_support_threshold, margin=a.ds_margin_threshold,
                    uncertainty=a.ds_uncertainty_threshold, conflict=a.ds_conflict_threshold)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    # Lazy imports keep --help and NumPy tests usable without torch/torchvision.
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    import Networks
    from dataset import RafDataSet, FER
    from prototype_utils import FeatureHook
    from train import build_transforms, calculate_target_thresholds
    config = config_from_args(args)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory is not empty; choose a new --output_dir')
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    weight_key = args.weights
    if weight_key == 'auto':
        weight_key = 'ema_teacher' if 'ema_teacher' in checkpoint else 'model'
    if weight_key not in checkpoint:
        raise KeyError('Checkpoint does not contain '+weight_key)
    model = Networks.Model(backbone=args.backbone, num_classes=7, pretrained=False).to(device)
    model.load_state_dict(checkpoint[weight_key], strict=True)
    model.eval()
    hook = FeatureHook(model.feature)
    weak, _, test = build_transforms()
    source = RafDataSet(args.source_path, phase='train', transform=test, basic_aug=False)
    threshold_data = FER(args.target_path, phase='train', transform=weak, basic_aug=False)
    target = FER(args.target_path, phase='train', transform=weak, weak2_transform=weak,
                 basic_aug=False, return_index=True)
    if threshold_data.file_paths != target.file_paths:
        raise RuntimeError('Target dataset IDs do not align')
    def loader(dataset):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # Dataset construction can affect RNG state; reset before inference.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    try:
        sx, sy = [], []
        with torch.inference_mode():
            for imgs, labels in loader(source):
                model(imgs.to(device), None, None, mode='test', task='source')
                sx.append(hook.output.detach().cpu().numpy())
                sy.append(labels.numpy())
        x, labels = np.concatenate(sx), np.concatenate(sy)
        bank = EvidenceBank(x, labels, np.arange(len(labels)), np.ones(len(labels)), 7)
        means = np.zeros((7, x.shape[1]), dtype=np.float32)
        radii = np.zeros(7, dtype=np.float32)
        ready = np.zeros(7, dtype=bool)
        for c in range(7):
            z = bank.x[bank.labels == c]
            if len(z):
                ready[c] = True
                means[c] = z.astype(np.float64).mean(0).astype(np.float32)
                radii[c] = np.quantile(np.linalg.norm(z-means[c], axis=1), args.source_radius_quantile)
        thresholds, _, global_mean, center = calculate_target_thresholds(
            model, loader(threshold_data), 7, args.threshold_base, args.threshold_beta,
            args.threshold_margin, args.threshold_min, args.threshold_max)
        ids = np.sort(np.random.default_rng(args.seed).choice(len(target), min(args.max_samples,len(target)), replace=False))
        rows, neighbors = [], []
        with torch.inference_mode():
            for view1, view2, _ignored_truth, sample_ids in loader(Subset(target, ids.tolist())):
                logits1, _ = model(view1.to(device), None, None, mode='test', task='target')
                f1 = hook.output.detach().clone()
                logits2, _ = model(view2.to(device), None, None, mode='test', task='target')
                f2 = hook.output.detach()
                z = F.normalize(F.normalize(f1,dim=1)+F.normalize(f2,dim=1),dim=1).cpu().numpy()
                conf1, pred1 = logits1.softmax(1).max(1)
                conf2, pred2 = logits2.softmax(1).max(1)
                evidence = bank.evaluate(z, sample_ids.numpy(), config)
                result = audit_source(evidence, pred1.cpu().numpy(), config)
                for j, sample_id in enumerate(sample_ids.tolist()):
                    y = int(pred1[j]); agreement = bool(pred1[j] == pred2[j])
                    confident = agreement and float(conf1[j]) >= float(thresholds[y]) and float(conf2[j]) >= float(thresholds[y])
                    distance = float(np.linalg.norm(z[j]-means[y]))
                    region = bool(ready[y] and np.isfinite(z[j]).all() and np.linalg.norm(z[j])>1e-8 and distance < radii[y])
                    candidate = agreement and not confident and region
                    detail, ns = diagnose_query(bank, z[j], y, sample_id, config)
                    for n in ns:
                        n['source_path'] = source.file_paths[n['source_id']]
                    neighbors.extend(ns)
                    rows.append(dict(sample_id=sample_id, target_path=target.file_paths[sample_id], pseudo_label=y,
                                     conf1=float(conf1[j]), conf2=float(conf2[j]), threshold=float(thresholds[y]),
                                     agreement=agreement, high_confidence=confident, region_ok=region,
                                     prototype_distance=distance, class_radius=float(radii[y]), rescue_candidate=candidate,
                                     source_available=bool(evidence['available'][j]), ds_support=float(result['support'][j]),
                                     ds_margin=float(result['margin'][j]), ds_unknown=float(result['uncertainty'][j]),
                                     ds_conflict=float(evidence['conflict'][j]), ds_pass=bool(result['pass_mask'][j]),
                                     accepted_if_rescue_enabled=confident or (candidate and bool(result['pass_mask'][j])), **detail))
        candidates = [r for r in rows if r['rescue_candidate']]
        summary = dict(checkpoint=str(Path(args.checkpoint).resolve()), weights=weight_key,
                       checkpoint_epoch=checkpoint.get('epoch'), options=vars(args), source_refs=len(bank.x),
                       target_samples=len(rows), sampled_ids=ids.tolist(), thresholds=thresholds.tolist(),
                       global_class_mean=global_mean, threshold_center=center,
                       rescue_candidates=len(candidates), candidates_ds_passed=sum(r['ds_pass'] for r in candidates),
                       candidate_support_min=min((r['ds_support'] for r in candidates),default=None),
                       candidate_support_ge_0999=sum(r['ds_support']>=.999 for r in candidates),
                       note='One checkpoint snapshot; no optimizer, backward, training or historical epoch replay. '
                            'Targets sampled before filtering. K-sweep masses are raw fusion; availability accounts for min_neighbors. '
                            'All targets are diagnosed, but only rescue_candidate rows would reach DS in training. '
                            'Fresh weak augmentations and recomputed thresholds/radii do not reproduce a historical training batch.')
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out/'queries.csv', rows)
        if neighbors:
            write_csv(out/'neighbors.csv', neighbors)
        else:
            (out/'neighbors.csv').write_text('sample_id,rank,source_id,neighbor_label,distance,quality,strength,strength_capped,same_label,source_path\n')
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:summary[k] for k in ['weights','source_refs','target_samples','rescue_candidates','candidates_ds_passed']},indent=2))
        print('Diagnostics saved:',out.resolve())
    finally:
        hook.close()


if __name__ == '__main__':
    run(parse_args())
