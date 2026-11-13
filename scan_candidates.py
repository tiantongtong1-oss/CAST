"""Fixed-checkpoint candidate scan; no optimizer, gradients or bank updates."""
import argparse
import csv
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
import Networks
from dataset import RafDataSet, FER
from prototype_utils import FeatureHook
from fused_knn import FusedKNNBank
from summarize_candidate_scan import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source_path', required=True)
    p.add_argument('--target_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--backbone', default='mobilenet_v2', choices=['mobilenet_v2', 'resnet18', 'resnet50'])
    p.add_argument('--weights', default='auto', choices=['auto', 'model', 'ema_teacher'])
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--k', type=int, default=11)
    p.add_argument('--query_chunk', type=int, default=64)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if min(args.k, args.batch_size, args.query_chunk) < 1 or args.workers < 0:
        p.error('k/batch_size/query_chunk must be positive; workers nonnegative')
    torch.manual_seed(1314)
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((256, 256)),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    source = RafDataSet(args.source_path, phase='train', transform=transform, basic_aug=False)
    target = FER(args.target_path, phase='train', transform=transform, basic_aug=False, return_index=True)
    def loader(ds):
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=device.type == 'cuda')
    model = Networks.Model(backbone=args.backbone, num_classes=7).to(device)
    # Only load trusted checkpoints. Explicit False supports newer PyTorch defaults.
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    key = args.weights
    if key == 'auto':
        key = 'ema_teacher' if 'ema_teacher' in checkpoint else 'model'
    if key not in checkpoint:
        raise ValueError('Checkpoint missing requested weights: ' + key)
    model.load_state_dict(checkpoint[key], strict=True)
    model.eval()
    hook = FeatureHook(model.feature)
    bank = FusedKNNBank(model.fc.in_features, 7, k=args.k).to(device)
    with torch.inference_mode():
        features, labels = [], []
        for images, y in loader(source):
            model(images.to(device), None, None, mode='test', task='source')
            features.append(hook.output.detach().cpu())
            labels.append(y)
        if not features:
            raise ValueError('Empty source dataset')
        bank.set_source_memory(torch.cat(features), torch.cat(labels))
        if not bank.class_center_ready.all():
            raise ValueError('All seven classes require at least two valid source features')
        if bank.source_labels.numel() < args.k:
            raise ValueError('Not enough valid source rows for k')
        fields = ['sample_id', 'path', 'true_label', 'teacher_pred', 'medoid_pred',
                  'source_knn_pred', 'balanced_knn_pred', 'teacher_top2', 'teacher_top3',
                  'medoid_top2', 'source_knn_top2', 'balanced_knn_top2']
        for prefix in ['teacher_p', 'medoid_d', 'source_knn_p', 'balanced_knn_d']:
            fields += [prefix + str(c) for c in range(7)]
        path = out / 'candidates.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for images, truth, ids in loader(target):
                logits, _ = model(images.to(device), None, None, mode='test', task='target')
                probs = logits.softmax(1)
                raw = hook.output.detach()
                if not torch.isfinite(raw).all() or (raw.norm(dim=1) <= 1e-8).any():
                    raise ValueError('Invalid target features; scan aborted rather than assigning candidates')
                z = F.normalize(raw, dim=1)
                md = torch.cdist(z.double(), bank.class_medoids.double())
                for start in range(0, len(ids), args.query_chunk):
                    stop = min(start + args.query_chunk, len(ids))
                    d = torch.cdist(z[start:stop].double(), bank.source_features.double())
                    nn = d.topk(args.k, largest=False, dim=1).indices
                    votes = F.one_hot(bank.source_labels[nn], 7).float().mean(1)
                    balanced = torch.stack([
                        d[:, bank.source_labels == c].topk(
                            min(args.k, int((bank.source_labels == c).sum())),
                            largest=False, dim=1).values.mean(1) for c in range(7)
                    ], dim=1)
                    for j in range(stop - start):
                        i = start + j
                        tr = probs[i].argsort(descending=True).tolist()
                        mr = md[i].argsort().tolist()
                        sr = votes[j].argsort(descending=True).tolist()
                        br = balanced[j].argsort().tolist()
                        row = dict(sample_id=int(ids[i]), path=target.file_paths[int(ids[i])],
                            true_label=int(truth[i]), teacher_pred=tr[0], medoid_pred=mr[0],
                            source_knn_pred=sr[0], balanced_knn_pred=br[0],
                            teacher_top2=json.dumps(tr[:2]), teacher_top3=json.dumps(tr[:3]),
                            medoid_top2=json.dumps(mr[:2]), source_knn_top2=json.dumps(sr[:2]),
                            balanced_knn_top2=json.dumps(br[:2]))
                        for prefix, values in [('teacher_p', probs[i]), ('medoid_d', md[i]),
                                               ('source_knn_p', votes[j]), ('balanced_knn_d', balanced[j])]:
                            row.update({prefix + str(c): float(values[c]) for c in range(7)})
                        writer.writerow(row)
                print('Scanned target through sample', int(ids[-1]) + 1, '/', len(target), flush=True)
    metadata = dict(vars(args), selected_weights=key, checkpoint_epoch=checkpoint.get('epoch'),
                    source_count=int(bank.source_labels.numel()), target_count=len(target),
                    scope='ALL target train samples; single deterministic view; not historical low-confidence subset',
                    label_order=['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral'])
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    summary = summarize(path)
    (out / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)
    print('Saved:', out.resolve())


if __name__ == '__main__':
    main()
