"""Read-only admission diagnostics. Never returns a training mask or updates banks."""
import csv
import os
import torch
import torch.nn.functional as F


@torch.no_grad()
def record_outside_radius(bank, features, pseudo, confidence, agreement,
                          sample_ids, logits1, logits2, epoch, output_dir,
                          truth=None):
    device = features.device
    labels = pseudo.long()
    safe = labels.clamp(0, bank.num_classes - 1)
    valid = torch.isfinite(features).all(1) & (features.norm(dim=1) > 1e-8)
    valid &= (labels >= 0) & (labels < bank.num_classes)
    valid &= bank.class_center_ready[safe] & bank.radius_ready
    low = agreement.bool() & ~confidence.bool()
    inside = bank.radius_accepts(features, labels)
    outside = low & valid & ~inside
    passed, checked, support, source, target, conflict = bank.select(
        features, labels, outside, agreement, sample_ids)
    z = F.normalize(torch.nan_to_num(features), dim=1)
    distance = (z.double() - bank.class_medoids[safe].double()).norm(dim=1)
    p1, p2 = logits1.softmax(1), logits2.softmax(1)
    mean = (p1 + p2) / 2
    js = ((p1 * (p1.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(1)
          + (p2 * (p2.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(1)) / 2
    os.makedirs(output_dir, exist_ok=True)
    # Caller reserves a fresh directory to avoid mixing different runs.
    path = os.path.join(output_dir, 'epoch_%03d.csv' % epoch)
    fields = ['epoch', 'sample_id', 'pseudo_label', 'region', 'confidence1',
              'confidence2', 'distance_over_R', 'knn_checked', 'knn_pass',
              'source_support', 'target_support', 'fused_support', 'domain_conflict', 'view_js']
    if truth is not None:
        fields += ['true_label', 'correct']
    columns = [sample_ids, labels, p1.max(1).values, p2.max(1).values,
               distance / bank.global_radius.double().clamp_min(1e-12),
               checked, passed, source, target, support, conflict, js]
    columns = [t.detach().cpu().tolist() for t in columns]
    low_cpu, valid_cpu, inside_cpu = [t.cpu().tolist() for t in (low, valid, inside)]
    truth_cpu = truth.detach().cpu().tolist() if truth is not None else None
    # One append per batch; caller reserves a new output directory per run.
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for j in range(len(low_cpu)):
            if not low_cpu[j]:
                continue
            region = 'invalid' if not valid_cpu[j] else ('inside' if inside_cpu[j] else 'outside')
            values = [epoch, columns[0][j], columns[1][j], region] + [c[j] for c in columns[2:]]
            row = dict(zip(fields, values))
            if truth_cpu is not None:
                row.update(true_label=truth_cpu[j], correct=int(truth_cpu[j] == columns[1][j]))
            writer.writerow(row)
