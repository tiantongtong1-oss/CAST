"""Source-only cRT-style experiment: fixed deterministic features/BN, train fc only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import Networks
from dataset import RafDataSet


def stratified_split(labels, fraction, seed):
    rng = random.Random(seed)
    train, val = [], []
    for c in range(7):
        ids = [i for i, y in enumerate(labels) if int(y) == c]
        if len(ids) < 2:
            raise ValueError('Each source class needs at least two samples')
        rng.shuffle(ids)
        n = max(1, min(len(ids)-1, round(len(ids)*fraction)))
        val.extend(ids[:n]); train.extend(ids[n:])
    return train, val


def metrics(truth, prediction):
    cm = torch.bincount(truth.cpu()*7+prediction.cpu(), minlength=49).reshape(7, 7).double()
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp_min(1)
    recall = tp / cm.sum(1).clamp_min(1)
    f1 = 2*precision*recall/(precision+recall).clamp_min(1e-12)
    return dict(accuracy=float(tp.sum()/cm.sum().clamp_min(1)),
                macro_f1=float(f1.mean()), uar=float(recall.mean()),
                precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist())


def frozen_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith('fc.'):
            h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--backbone', default='mobilenet_v2', choices=['mobilenet_v2','resnet18','resnet50'])
    p.add_argument('--weights', default='model', choices=['model','ema_teacher'])
    p.add_argument('--head_init', default='reset', choices=['reset','keep'])
    p.add_argument('--sampling', default='balanced', choices=['balanced','instance'])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=1314)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.batch_size < 1 or args.workers < 0 or not 0 < args.val_fraction < 1:
        p.error('Invalid epoch/lr/batch/workers/val_fraction')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((256,256)), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds = RafDataSet(args.source_path, phase='train', transform=transform, basic_aug=False)
    train_ids, val_ids = stratified_split(ds.label, args.val_fraction, args.seed)
    (out/'split.json').write_text(json.dumps(dict(
        train_paths=[ds.file_paths[i] for i in train_ids],
        val_paths=[ds.file_paths[i] for i in val_ids]), indent=2))
    model = Networks.Model(backbone=args.backbone, num_classes=7, pretrained=False).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint[args.weights], strict=True)
    for param in model.parameters(): param.requires_grad_(False)
    model.eval()  # Includes feature Dropout and both feature/output BN.
    digest = frozen_digest(model)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type=='cuda')
    features, labels = [], []
    with torch.no_grad():
        for imgs, y in loader:
            f = model.feature(imgs.to(device))
            if not torch.isfinite(f).all(): raise ValueError('Nonfinite source features')
            features.append(f.cpu()); labels.append(y)
    x, y = torch.cat(features), torch.cat(labels).long()
    def evaluate_head(ids):
        preds = []
        with torch.no_grad():
            for start in range(0, len(ids), args.batch_size):
                f = x[ids[start:start+args.batch_size]].to(device)
                preds.append(model.bn(model.fc(f)).argmax(1).cpu())
        return metrics(y[ids], torch.cat(preds))
    baseline = evaluate_head(val_ids)
    (out/'baseline_source_val.json').write_text(json.dumps(baseline, indent=2))
    if args.head_init == 'reset': model.fc.reset_parameters()
    for param in model.fc.parameters(): param.requires_grad_(True)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_y = y[train_ids]
    counts = torch.bincount(train_y, minlength=7)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = (torch.utils.data.WeightedRandomSampler(
        counts.double().reciprocal()[train_y], len(train_ids), replacement=True, generator=generator)
        if args.sampling=='balanced' else torch.utils.data.RandomSampler(
            train_ids, generator=generator))
    train_set = torch.utils.data.TensorDataset(x[train_ids], train_y)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    criterion = nn.CrossEntropyLoss()
    best = -1.0
    with (out/'metrics.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['epoch','loss','source_val_accuracy','source_val_macro_f1','source_val_uar'])
        writer.writeheader()
        for epoch in range(1,args.epochs+1):
            model.eval()  # Never call model.train(); frozen BN/Dropout must stay fixed.
            total, n = 0.0, 0
            for f, target in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits=model.bn(model.fc(f.to(device)))
                loss=criterion(logits,target.to(device)); loss.backward(); optimizer.step()
                total+=loss.item()*len(target); n+=len(target)
            scheduler.step()
            result=evaluate_head(val_ids)
            if frozen_digest(model)!=digest: raise RuntimeError('Non-fc state changed')
            state=dict(model=model.state_dict(),epoch=epoch,args=vars(args),
                source_val=result,source_train_counts=counts.tolist(),
                selected_weights=args.weights,frozen_state_sha256=digest)
            torch.save(state,out/'head_last.pth')
            if result['macro_f1']>best:
                best=result['macro_f1']; torch.save(state,out/'head_best.pth')
                (out/'best_source_val.json').write_text(json.dumps(result,indent=2))
            row=dict(epoch=epoch,loss=total/n,source_val_accuracy=result['accuracy'],
                     source_val_macro_f1=result['macro_f1'],source_val_uar=result['uar'])
            writer.writerow(row);handle.flush(); print(json.dumps(row),flush=True)
    print('Saved:',out.resolve(),'best source-val Macro-F1:',best)


if __name__=='__main__': main()
