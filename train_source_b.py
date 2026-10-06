"""Experiment B: source CE/affinity + single-prototype margin; RAF validation selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import Networks
from dataset import RafDataSet
from train import build_transforms, classifier_modulation_loss, seed, _init_fn
from prototype_utils import FeatureHook
from pseudo_audit import PredictionAudit
from source_margin import prototype_margin_loss


def stratified_split(labels, fraction, split_seed):
    rng = np.random.RandomState(split_seed)
    train, val = [], []
    for c in range(7):
        ids = np.flatnonzero(np.asarray(labels) == c)
        if len(ids) < 2:
            raise ValueError('each source class needs at least two samples')
        rng.shuffle(ids)
        n = max(1, min(len(ids)-1, int(round(len(ids)*fraction))))
        val.extend(ids[:n]); train.extend(ids[n:])
    return sorted(train), sorted(val)


@torch.no_grad()
def evaluate_source(model, loader, name):
    model.eval()
    audit = PredictionAudit()
    for images, labels in loader:
        logits, _ = model(images.cuda(non_blocking=True), None, None, mode='test', task='source')
        audit.add(labels.numpy(), logits.argmax(1).cpu().numpy())
    audit.report(name)
    cm = audit.matrices['all']
    tp = cm.diagonal().astype(float)
    f1 = np.divide(2*tp, cm.sum(0)+cm.sum(1), out=np.zeros(7),
                   where=(cm.sum(0)+cm.sum(1))>0)
    print('[SOURCE F1] per_class=%s disgust_F1=%.6f' % (f1.tolist(), f1[2]), flush=True)
    return float(f1.mean())


@torch.no_grad()
def refresh_prototypes(model, hook, loader):
    model.eval()
    sums = torch.zeros(7, model.fc.in_features, device='cuda')
    counts = torch.zeros(7, device='cuda')
    for images, labels in loader:
        labels = labels.cuda(non_blocking=True)
        model(images.cuda(non_blocking=True), None, None, mode='test', task='source')
        features = torch.nn.functional.normalize(hook.output, dim=1)
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    if torch.any(counts == 0):
        raise ValueError('prototype training split has missing classes')
    return torch.nn.functional.normalize(sums/counts[:, None], dim=1).detach()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_path', default='/workspace/ttt/code/test-upload-clean/datesets/raf-basic')
    p.add_argument('--backbone', default='mobilenet_v2', choices=['mobilenet_v2','resnet18','resnet50'])
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--val_fraction', type=float, default=0.2)
    p.add_argument('--split_seed', type=int, default=2000)
    p.add_argument('--margin', type=float, default=0.1)
    p.add_argument('--margin_weight', type=float, default=0.1, help='0 gives paired source baseline A')
    p.add_argument('--w1', type=float, default=4.0)
    p.add_argument('--w2', type=float, default=0.3)
    p.add_argument('--w3', type=float, default=0.1)
    p.add_argument('--run_name', default='source_B_single_prototype_margin')
    a = p.parse_args()
    if not 0 < a.val_fraction < 1 or a.epochs < 1 or a.batch_size < 2 or a.lr <= 0:
        p.error('invalid split/training settings')
    if not 0 <= a.margin <= 2 or a.margin_weight < 0 or a.workers < 0:
        p.error('invalid margin/worker settings')
    if Path(a.run_name).name != a.run_name or a.run_name in {'.','..'}:
        p.error('run_name must be one directory name')
    out = Path('new_models')/'rafdb_fer'/a.run_name
    out.mkdir(parents=True, exist_ok=False)
    weak, _, fixed = build_transforms()
    train_data = RafDataSet(a.source_path, 'train', transform=weak, basic_aug=False)
    fixed_data = RafDataSet(a.source_path, 'train', transform=fixed, basic_aug=False)
    if train_data.file_paths != fixed_data.file_paths or not np.array_equal(train_data.label, fixed_data.label):
        raise ValueError('training/fixed source dataset paths or labels do not align')
    train_ids, val_ids = stratified_split(train_data.label, a.val_fraction, a.split_seed)
    manifest = dict(seed=seed, split_seed=a.split_seed,
                    initialized_checkpoint=a.checkpoint,
                    validation_previously_seen_warning=bool(a.checkpoint),
                    train_paths=[train_data.file_paths[i] for i in train_ids],
                    validation_paths=[train_data.file_paths[i] for i in val_ids])
    (out/'split.json').write_text(json.dumps(manifest, indent=2))
    if a.checkpoint:
        print('DIAGNOSTIC FINETUNE: validation may have been seen by the supplied checkpoint.', flush=True)
    def loader(dataset, ids, shuffle=False):
        return DataLoader(Subset(dataset, ids), batch_size=a.batch_size,
                          shuffle=shuffle, num_workers=a.workers, pin_memory=True,
                          worker_init_fn=_init_fn)
    train_loader = loader(train_data, train_ids, True)
    proto_loader = loader(fixed_data, train_ids)
    val_loader = loader(fixed_data, val_ids)
    model = Networks.Model(backbone=a.backbone, num_classes=7).cuda()
    if a.checkpoint:
        model.load_state_dict(torch.load(a.checkpoint, map_location='cuda')['model'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    hook = FeatureHook(model.feature)
    best = -1.0
    best_path = out/'source_B_best_macro_f1.pth'
    print('Experiment B: weight=%g margin=%g train=%d val=%d output=%s' %
          (a.margin_weight, a.margin, len(train_ids), len(val_ids), out), flush=True)
    try:
        for epoch in range(a.epochs):
            prototypes = refresh_prototypes(model, hook, proto_loader) if a.margin_weight else None
            model.train()
            totals = np.zeros(4); batches = 0
            for images, labels in train_loader:
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                optimizer.zero_grad()
                result = model(images, labels, None, 'train', 'source')
                ce = torch.nn.functional.cross_entropy(result[0], labels)
                affinity = result[1]
                weight = classifier_modulation_loss(model)
                margin_loss = (prototype_margin_loss(hook.output, labels, prototypes, a.margin)
                               if prototypes is not None else ce.new_zeros(()))
                loss = a.w1*ce + a.w2*affinity + a.w3*weight + a.margin_weight*margin_loss
                loss.backward(); optimizer.step()
                totals += [ce.item(), affinity.item(), weight.item(), margin_loss.item()]
                batches += 1
            scheduler.step()
            print('[B Epoch %d] CE/Affinity/Weight/Margin=%s' % (epoch, totals/max(batches,1)), flush=True)
            score = evaluate_source(model, val_loader, 'RAFDB/validation/B epoch %d' % epoch)
            if score > best:
                best = score
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(), epoch=epoch,
                                best_source_val_macro_f1=best, args=vars(a),
                                source_prototypes=prototypes, split_file=str(out/'split.json')), best_path)
                print('Saved best SOURCE validation macro-F1 %.6f: %s' % (best, best_path), flush=True)
        model.load_state_dict(torch.load(best_path, map_location='cuda')['model'])
        # Test is consulted only after checkpoint selection is complete.
        test = RafDataSet(a.source_path, 'test', transform=fixed, basic_aug=False)
        test_loader = DataLoader(test, batch_size=a.batch_size, shuffle=False,
                                 num_workers=a.workers, pin_memory=True)
        evaluate_source(model, test_loader, 'RAFDB/test/selected B')
    finally:
        hook.close()


if __name__ == '__main__':
    main()
