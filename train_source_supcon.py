"""Source-only balanced-batch supervised contrastive fine-tuning (not full BCL)."""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import Networks
from dataset import RafDataSet
from retrain_balanced_head import stratified_split, metrics


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, indices, per_class=8, seed=1314):
        self.groups = [[i for i in indices if int(labels[i]) == c] for c in range(7)]
        if any(not g for g in self.groups): raise ValueError('Missing source train class')
        self.per_class, self.seed, self.epoch = per_class, seed, 0
        self.steps = math.ceil(len(indices)/(7*per_class))
    def __len__(self): return self.steps
    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        for _ in range(self.steps):
            batch=[]
            for g in self.groups:
                batch.extend(rng.sample(g,self.per_class) if len(g)>=self.per_class else rng.choices(g,k=self.per_class))
            rng.shuffle(batch)
            yield batch


def supervised_contrastive(z, labels, temperature=.1):
    z=F.normalize(z,dim=1)
    logits=z @ z.t()/temperature
    diagonal=torch.eye(len(z),device=z.device,dtype=torch.bool)
    positives=labels[:,None].eq(labels[None,:]) & ~diagonal
    log_probs=logits-torch.logsumexp(logits.masked_fill(diagonal,-torch.inf),dim=1,keepdim=True)
    valid=positives.sum(1)>0
    loss=-(log_probs.masked_fill(~positives,0).sum(1)/positives.sum(1).clamp_min(1))
    return loss[valid].mean() if valid.any() else z.sum()*0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--source_path',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--backbone',default='mobilenet_v2',choices=['mobilenet_v2','resnet18','resnet50'])
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--contrast_weight',type=float,default=.1)
    p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--per_class',type=int,default=8)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=1314)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if min(args.epochs,args.per_class,args.lr,args.temperature)<=0 or args.contrast_weight<0 or args.workers<0:
        p.error('Invalid training arguments')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device)
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    weak=transforms.Compose([transforms.ToPILImage(),transforms.Resize((256,256)),
        transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),norm])
    fixed=transforms.Compose([transforms.ToPILImage(),transforms.Resize((256,256)),
        transforms.CenterCrop(224),transforms.ToTensor(),norm])
    train=RafDataSet(args.source_path,'train',transform=weak,strong_transform=weak,basic_aug=False)
    val=RafDataSet(args.source_path,'train',transform=fixed,basic_aug=False)
    assert train.file_paths==val.file_paths and np.array_equal(train.label,val.label)
    train_ids,val_ids=stratified_split(train.label,.1,args.seed)
    (out/'split.json').write_text(json.dumps(dict(train_paths=[train.file_paths[i] for i in train_ids],val_paths=[val.file_paths[i] for i in val_ids]),indent=2))
    sampler=BalancedBatchSampler(train.label,train_ids,args.per_class,args.seed)
    loader=torch.utils.data.DataLoader(train,batch_sampler=sampler,num_workers=args.workers)
    val_loader=torch.utils.data.DataLoader(torch.utils.data.Subset(val,val_ids),batch_size=128,num_workers=args.workers)
    model=Networks.Model(backbone=args.backbone,num_classes=7,pretrained=False).to(device)
    try: ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    except TypeError: ckpt=torch.load(args.checkpoint,map_location='cpu')
    model.load_state_dict(ckpt['model'],strict=True)
    projection=nn.Sequential(nn.Linear(model.fc.in_features,128),nn.ReLU(),nn.Linear(128,128)).to(device)
    optimizer=torch.optim.Adam(list(model.parameters())+list(projection.parameters()),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    def evaluate():
        model.eval();truth=[];pred=[]
        with torch.no_grad():
            for imgs,y in val_loader:
                features=model.feature(imgs.to(device));logits=model.bn(model.fc(features))
                truth.append(y);pred.append(logits.argmax(1).cpu())
        return metrics(torch.cat(truth),torch.cat(pred))
    (out/'baseline_source_val.json').write_text(json.dumps(evaluate(),indent=2))
    best=-1
    with (out/'metrics.jsonl').open('w') as log:
        for epoch in range(1,args.epochs+1):
            sampler.epoch=epoch-1;model.train();projection.train()
            ce_total=con_total=0.;steps=0
            for x1,x2,y in loader:
                labels=torch.cat([y,y]).to(device)
                features=model.feature(torch.cat([x1,x2]).to(device))
                logits=model.bn(model.fc(features))
                ce=F.cross_entropy(logits,labels)
                # Projection executes in both conditions to match model RNG usage.
                z=projection(features)
                con=supervised_contrastive(z,labels,args.temperature)
                loss=ce+args.contrast_weight*con
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                ce_total+=ce.item();con_total+=con.item();steps+=1
            scheduler.step();result=evaluate()
            state=dict(model=model.state_dict(),projection=projection.state_dict(),epoch=epoch,args=vars(args),source_val=result)
            torch.save(state,out/'source_last.pth')
            if result['macro_f1']>best:
                best=result['macro_f1'];torch.save(state,out/'source_best.pth')
                (out/'best_source_val.json').write_text(json.dumps(result,indent=2))
            row=dict(epoch=epoch,ce=ce_total/steps,contrast=con_total/steps,source_val=result)
            text=json.dumps(row);log.write(text+'\n');log.flush();print(text,flush=True)
    print('Saved:',out.resolve())


if __name__=='__main__':main()
