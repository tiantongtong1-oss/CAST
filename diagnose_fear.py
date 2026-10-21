"""Read-only FER validation diagnosis; no fitting, threshold tuning or model writes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

CLASSES = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']


def summarize(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), 7) or not len(labels):
        raise ValueError('expected nonempty N x 7 probabilities')
    if ((labels < 0) | (labels >= 7)).any() or not np.isfinite(probabilities).all():
        raise ValueError('invalid labels/probabilities')
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError('probabilities must be normalized')
    prediction = probabilities.argmax(1)
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (labels, prediction), 1)
    # Stable class-index tie breaking, matching argmax for top-1.
    order = np.argsort(-probabilities, axis=1, kind='stable')
    ranks = np.argmax(order == 1, axis=1) + 1
    fear = labels == 1
    missed = fear & (prediction != 1)
    gap = probabilities.max(1) - probabilities[:, 1]
    tp = int(matrix[1, 1])
    precision = tp / max(int(matrix[:, 1].sum()), 1)
    recall = tp / max(int(fear.sum()), 1)
    stats = dict(samples=len(labels), accuracy=float((prediction == labels).mean()),
        fear_support=int(fear.sum()), fear_predicted=int((prediction == 1).sum()),
        fear_correct=tp, fear_precision=precision, fear_recall=recall,
        fear_f1=2*precision*recall/max(precision+recall, 1e-12),
        fear_rank_counts={str(i):int((fear & (ranks == i)).sum()) for i in range(1,8)},
        fear_top2_recall=float((ranks[fear] <= 2).mean()) if fear.any() else None,
        fear_top3_recall=float((ranks[fear] <= 3).mean()) if fear.any() else None,
        fear_prediction_destinations=dict(zip(CLASSES, matrix[1].tolist())),
        missed_fear_probability_gap_quantiles=(dict(zip(['p05','p50','p95'],
            np.quantile(gap[missed], [.05,.5,.95]).tolist())) if missed.any() else None),
        tie_policy='descending probability, ascending class index')
    return matrix, prediction, ranks, gap, stats


def write_reports(output, paths, labels, probabilities, metadata):
    matrix, prediction, ranks, gap, stats = summarize(labels, probabilities)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [('confusion_counts.csv', matrix),
                         ('confusion_row_normalized.csv', matrix / np.maximum(matrix.sum(1, keepdims=True), 1))]:
        with (output / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['true_class / predicted_class'] + CLASSES)
            writer.writerows([CLASSES[i]] + row.tolist() for i,row in enumerate(values))
    header = ['path','true_class','predicted_class','fear_rank','fear_probability',
              'top1_minus_fear'] + ['prob_'+c for c in CLASSES]
    with (output/'predictions.csv').open('w', newline='', encoding='utf-8') as all_file, \
         (output/'fear_predictions.csv').open('w', newline='', encoding='utf-8') as fear_file:
        writers = [csv.writer(all_file), csv.writer(fear_file)]
        for writer in writers:
            writer.writerow(header)
        for i, label in enumerate(labels):
            row = [paths[i], CLASSES[label], CLASSES[prediction[i]], int(ranks[i]),
                   float(probabilities[i,1]), float(gap[i])] + probabilities[i].tolist()
            writers[0].writerow(row)
            if label == 1:
                writers[1].writerow(row)
    stats['metadata'] = metadata
    (output/'summary.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--target_path', required=True)
    parser.add_argument('--backbone', default='mobilenet_v2', choices=['mobilenet_v2','resnet18','resnet50'])
    parser.add_argument('--weights', choices=['model','ema_teacher'], default='model')
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch_size must be positive')
    # Import training preprocessing only after argument parsing; never run training.
    import torch
    import Networks
    from dataset import FER
    from train import build_transforms
    device = torch.device(args.device)
    _, _, transform = build_transforms()
    dataset = FER(args.target_path, phase='val', transform=transform, basic_aug=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.weights not in checkpoint:
        raise KeyError('checkpoint does not contain '+args.weights)
    model = Networks.Model(backbone=args.backbone, num_classes=7).to(device)
    model.load_state_dict(checkpoint[args.weights], strict=True)
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for images, truth in loader:
            # Same call as train.evaluate: labels are not used by test-mode forward.
            logits, _ = model(images.to(device), truth.to(device), None, mode='test')
            labels.append(truth.numpy())
            probabilities.append(logits.softmax(1).cpu().numpy())
    write_reports(args.output_dir, dataset.file_paths, np.concatenate(labels),
                  np.concatenate(probabilities), dict(checkpoint=str(Path(args.checkpoint).resolve()),
                    weights=args.weights, split='val', backbone=args.backbone,
                    checkpoint_epoch=checkpoint.get('epoch'),
                    checkpoint_best_val_acc=checkpoint.get('best_val_acc')))


if __name__ == '__main__':
    main()
