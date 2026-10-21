"""Evaluate a checkpoint on the official RAF-DB test split; no training."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from diagnose_fear import CLASSES, summarize, write_reports


def class_metrics(matrix):
    matrix = np.asarray(matrix)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    correct = matrix.diagonal()
    recall = correct / np.maximum(support, 1)
    precision = correct / np.maximum(predicted, 1)
    f1 = 2 * recall * precision / np.maximum(recall + precision, 1e-12)
    rows = [dict(class_name=name, recall=float(recall[i]), precision=float(precision[i]),
                 f1=float(f1[i]), support=int(support[i]), predicted=int(predicted[i]),
                 correct=int(correct[i])) for i, name in enumerate(CLASSES)]
    return rows, float(recall[support > 0].mean()), float(f1.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source_path', required=True)
    p.add_argument('--backbone', choices=['mobilenet_v2', 'resnet18', 'resnet50'], default='mobilenet_v2')
    p.add_argument('--weights', choices=['model','ema_teacher'], default='model')
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--output_dir', required=True)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch_size must be positive')
    import torch
    import Networks
    from dataset import RafDataSet
    from train import build_transforms
    device = torch.device(a.device)
    _, _, transform = build_transforms()
    dataset = RafDataSet(a.source_path, phase='test', transform=transform,
                         strong_transform=None, basic_aug=False)
    if not len(dataset):
        raise ValueError('RAF-DB test split is empty')
    loader = torch.utils.data.DataLoader(dataset, batch_size=a.batch_size,
                                         shuffle=False, num_workers=0)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if a.weights not in checkpoint:
        raise KeyError('Missing checkpoint weights: '+a.weights)
    model = Networks.Model(backbone=a.backbone, num_classes=7).to(device)
    model.load_state_dict(checkpoint[a.weights], strict=True)
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for images, truth in loader:
            logits, _ = model(images.to(device), None, None, mode='test', task='source')
            labels.append(truth.numpy())
            probabilities.append(logits.softmax(1).cpu().numpy())
    labels, probabilities = np.concatenate(labels), np.concatenate(probabilities)
    output = Path(a.output_dir)
    metadata = dict(checkpoint=str(Path(a.checkpoint).resolve()), weights=a.weights,
        dataset='RAF-DB', split='test', backbone=a.backbone,
        checkpoint_epoch=checkpoint.get('epoch'),
        checkpoint_best_val_acc=checkpoint.get('best_val_acc'),
        note='checkpoint_best_val_acc is historical metadata, not RAF-DB test accuracy')
    write_reports(output, dataset.file_paths, labels, probabilities, metadata)
    matrix, _, _, _, stats = summarize(labels, probabilities)
    metrics, uar, macro_f1 = class_metrics(matrix)
    stats.update(metadata=metadata, per_class=metrics, uar_present_classes=uar, macro_f1=macro_f1)
    (output/'summary.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output/'class_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    print('\nRAF-DB TEST: accuracy=%.4f UAR=%.4f Macro-F1=%.4f' % (stats['accuracy'], uar, macro_f1))
    print('%-12s %9s %9s %9s %8s' % ('Class','Recall','Precision','F1','Support'))
    for row in metrics:
        print('%-12s %8.2f%% %8.2f%% %8.2f%% %8d' %
              (row['class_name'], 100*row['recall'], 100*row['precision'], 100*row['f1'], row['support']))
    print('Reports:', output.resolve())


if __name__ == '__main__':
    main()
