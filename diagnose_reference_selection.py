"""Independent frozen-source neighbor audit of unchanged Teacher selections."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

NAMES = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']


def classify_path(agreement, direct, radius, available, support, threshold):
    if direct:
        return 'direct'
    if agreement and radius and available and support >= threshold:
        return 'rescue'
    return 'rejected'


def reference_scores(mass, label, available, threshold):
    mass = np.asarray(mass, dtype=float)
    if mass.shape != (7,) or not np.isfinite(mass).all():
        raise ValueError('expected seven finite reference supports')
    other = mass.copy()
    other[label] = -np.inf
    top = int(mass.argmax())
    ordered = np.sort(mass)
    fields = dict(reference_available=bool(available),
        reference_support_pred=float(mass[label]),
        reference_pass=bool(available and mass[label] >= threshold),
        reference_contradiction=bool(available and other.max() > mass[label]),
        reference_top=NAMES[top] if available else 'unavailable',
        reference_top_margin=float(ordered[-1]-ordered[-2]),
        reference_pred_margin=float(mass[label]-other.max()),
        reference_fear_margin=float(mass[1]-np.max(np.delete(mass,1))))
    fields.update({'reference_support_'+name:float(mass[c]) for c,name in enumerate(NAMES)})
    return fields


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--reference_checkpoint', required=True)
    p.add_argument('--reference_threshold', type=float, default=.92)
    p.add_argument('--source_path', required=True)
    p.add_argument('--target_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--seed', type=int, default=101314)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--support_threshold', type=float, default=.92)
    p.add_argument('--k', type=int, default=11)
    p.add_argument('--min_neighbors', type=int, default=3)
    a = p.parse_args()
    if a.batch_size < 1 or not 0 <= a.support_threshold <= 1 or not 0 <= a.reference_threshold <= 1:
        p.error('invalid batch size or support threshold')
    import torch
    import torch.nn.functional as F
    import Networks
    from dataset import FER, RafDataSet
    from train import build_transforms, calculate_target_thresholds
    from calibration_rng import isolated_rng
    from source_prototype_radius import SourcePrototypeRadiusBank
    from dual_domain_ds import DSConfig
    from neighbor_support import MeanSupportBank
    from ema_utils import select_dual_view_pseudo_labels

    device = torch.device(a.device)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    saved = checkpoint.get('args', {})
    if 'ema_teacher' not in checkpoint:
        raise ValueError('Checkpoint must contain ema_teacher')
    model = Networks.Model(backbone=saved.get('backbone', 'mobilenet_v2'), num_classes=7).to(device)
    model.load_state_dict(checkpoint['ema_teacher'], strict=True)
    model.eval()
    reference_checkpoint = torch.load(a.reference_checkpoint, map_location='cpu', weights_only=False)
    reference_args = reference_checkpoint.get('args', {})
    reference = Networks.Model(backbone=reference_args.get('backbone', saved.get('backbone', 'mobilenet_v2')),
                               num_classes=7).to(device)
    reference.load_state_dict(reference_checkpoint['model'], strict=True)
    reference.eval()
    reference.requires_grad_(False)
    weak, _, test = build_transforms()
    source = RafDataSet(a.source_path, phase='train', transform=test, basic_aug=False)
    train = FER(a.target_path, phase='train', transform=weak, basic_aug=False)
    validation = FER(a.target_path, phase='val', transform=weak,
                     weak2_transform=weak, basic_aug=False, return_index=True)
    def loader(dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=a.batch_size,
                                          shuffle=False, num_workers=0)
    radius = SourcePrototypeRadiusBank(7, model.fc.in_features,
        source_radius_quantile=saved.get('source_radius_quantile', .9)).to(device)
    config = DSConfig(k=a.k, min_neighbors=a.min_neighbors)
    with isolated_rng(a.seed), torch.no_grad():
        print('Building source references...', flush=True)
        radius.begin_source_refresh()
        features, labels, reference_features = [], [], []
        for images, truth in loader(source):
            _, f = model(images.to(device), None, None, mode='test')
            radius.accumulate_source(f.to(device), truth.to(device))
            features.append(f.numpy())
            labels.append(truth.numpy())
            _, reference_f = reference(images.to(device), None, None, mode='test')
            reference_features.append(reference_f.numpy())
        radius.finalize_source()
        sy = np.concatenate(labels)
        bank = MeanSupportBank(np.concatenate(features), sy, np.arange(len(sy)), np.ones(len(sy)), 7)
        reference_bank = MeanSupportBank(np.concatenate(reference_features), sy, np.arange(len(sy)), np.ones(len(sy)), 7)
        print('Calibrating confidence on target TRAIN only...', flush=True)
        thresholds, _, _, _ = calculate_target_thresholds(model, loader(train), 7,
            saved.get('threshold_base', .85), saved.get('threshold_beta', .5),
            saved.get('threshold_margin', .02), saved.get('threshold_min', .8),
            saved.get('threshold_max', .95))
    rows = []
    # Separate seed means validation augmentations do not depend on calibration length.
    with isolated_rng(a.seed + 1), torch.no_grad():
        print('Auditing all validation samples, including direct accepts...', flush=True)
        for w1, w2, truth, ids in loader(validation):
            l1, f1 = model(w1.to(device), None, None, mode='test')
            l2, f2 = model(w2.to(device), None, None, mode='test')
            pseudo, direct, _ = select_dual_view_pseudo_labels(l1, l2, thresholds)
            pred2 = l2.argmax(1)
            z = F.normalize(F.normalize(f1.to(device), dim=1) + F.normalize(f2.to(device), dim=1), dim=1)
            region = radius.in_distribution(z, pseudo).cpu().numpy()
            evidence = bank.evaluate(z.cpu().numpy(), ids.numpy(), config)
            # Reuse the exact tensors: no additional augmentation calls or RNG consumption.
            _, rf1 = reference(w1.to(device), None, None, mode='test')
            _, rf2 = reference(w2.to(device), None, None, mode='test')
            rz = F.normalize(F.normalize(rf1, dim=1) + F.normalize(rf2, dim=1), dim=1)
            reference_evidence = reference_bank.evaluate(rz.numpy(), ids.numpy(), config)
            minimum = torch.minimum(l1.softmax(1).max(1).values, l2.softmax(1).max(1).values).cpu().numpy()
            pseudo, pred2, direct = pseudo.cpu().numpy(), pred2.cpu().numpy(), direct.bool().cpu().numpy()
            for j, sample_id in enumerate(ids.tolist()):
                label = int(pseudo[j])
                mass = evidence['mass'][j]
                support = float(mass[label])
                other = mass.copy(); other[label] = -np.inf
                available = bool(evidence['available'][j])
                agreement = bool(label == pred2[j])
                path = classify_path(agreement, bool(direct[j]), bool(region[j]), available, support, a.support_threshold)
                neighbor_pass = available and support >= a.support_threshold
                # Contradiction = a different class has strictly larger support, not merely a low score.
                contradiction = available and float(other.max()) > support
                ref_fields = reference_scores(reference_evidence['mass'][j], label,
                    bool(reference_evidence['available'][j]), a.reference_threshold)
                rows.append(dict(sample_id=sample_id, path=validation.file_paths[sample_id],
                    true_class=NAMES[int(truth[j])], pred1=NAMES[label], pred2=NAMES[int(pred2[j])],
                    agreement=agreement, min_confidence=float(minimum[j]),
                    confidence_threshold=float(thresholds[label]), direct=bool(direct[j]),
                    radius_pass=bool(region[j]), neighbor_count=int(evidence['count'][j]),
                    neighbor_available=available, neighbor_support=support,
                    neighbor_top=NAMES[int(mass.argmax())] if available else 'unavailable',
                    neighbor_margin=float(support-other.max()), neighbor_pass=neighbor_pass,
                    neighbor_contradiction=contradiction, selection_path=path,
                    **{'support_'+name:float(mass[c]) for c,name in enumerate(NAMES)}, **ref_fields))
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out/'selection_rows.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    def aggregate(group):
        return dict(count=len(group), direct=sum(r['direct'] for r in group),
            rescue=sum(r['selection_path']=='rescue' for r in group),
            rejected=sum(r['selection_path']=='rejected' for r in group),
            direct_neighbor_failed=sum(r['direct'] and not r['neighbor_pass'] for r in group),
            direct_neighbor_contradiction=sum(r['direct'] and r['neighbor_contradiction'] for r in group),
            direct_neighbor_top_fear=sum(r['direct'] and r['neighbor_top']=='Fear' for r in group),
            accepted=sum(r['selection_path']!='rejected' for r in group),
            accepted_reference_failed=sum(r['selection_path']!='rejected' and not r['reference_pass'] for r in group),
            accepted_reference_contradiction=sum(r['selection_path']!='rejected' and r['reference_contradiction'] for r in group),
            accepted_reference_top_fear=sum(r['selection_path']!='rejected' and r['reference_top']=='Fear' for r in group),
            accepted_reference_strong_fear=sum(r['selection_path']!='rejected' and r['reference_top']=='Fear'
                and r['reference_support_Fear']>=a.reference_threshold for r in group))
    fear = [r for r in rows if r['true_class']=='Fear']
    summary = dict(metadata=dict(checkpoint=str(Path(a.checkpoint).resolve()), epoch=checkpoint.get('epoch'),
        split='val', seed=a.seed, support_threshold=a.support_threshold, k=a.k,
        reference_checkpoint=str(Path(a.reference_checkpoint).resolve()), reference_threshold=a.reference_threshold,
        confidence_thresholds=thresholds.tolist(), radii=radius.class_radii.cpu().tolist(),
        note='Final Teacher snapshot simulation, not historical replay. Fixed support; no Otsu.'),
        all=aggregate(rows), by_true_class={name:aggregate([r for r in rows if r['true_class']==name]) for name in NAMES},
        fear_both_views_correct=aggregate([r for r in fear if r['pred1']==r['pred2']=='Fear']),
        fear_agreed_wrong=aggregate([r for r in fear if r['agreement'] and r['pred1']!='Fear']),
        accepted_correct=aggregate([r for r in rows if r['selection_path']!='rejected' and r['pred1']==r['true_class']]),
        accepted_wrong=aggregate([r for r in rows if r['selection_path']!='rejected' and r['pred1']!=r['true_class']]),
        fear_accepted_wrong=aggregate([r for r in fear if r['selection_path']!='rejected' and r['pred1']!='Fear']),
        fear_accepted_correct=aggregate([r for r in fear if r['selection_path']!='rejected' and r['pred1']=='Fear']),
        direct_correct=aggregate([r for r in rows if r['direct'] and r['pred1']==r['true_class']]),
        direct_wrong=aggregate([r for r in rows if r['direct'] and r['pred1']!=r['true_class']]))
    (out/'selection_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
