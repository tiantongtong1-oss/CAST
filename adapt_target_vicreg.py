"""Source supervised CE + target VICReg; no target pseudo-labels or alignment."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
from vicreg_adaptation_utils import discover_unlabeled_train,vicreg_loss
from topk_diagnostics import summarize
from feature_knn_diagnostics import classification_metrics
from diagnostic_data_audit import assert_disjoint_paths
from adaptation_stability import freeze_backbone_bn_statistics,shared_gradient_metrics


class UnlabeledTwoViews:
    def __init__(self,root,transform):
        self.paths=discover_unlabeled_train(root);self.transform=transform
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        import cv2
        image=cv2.imread(self.paths[index])
        if image is None:raise FileNotFoundError(self.paths[index])
        rgb=image[:,:,::-1].copy()
        return self.transform(rgb),self.transform(rgb)


def seed_worker(_):
    import torch
    worker_seed=torch.initial_seed()%2**32
    random.seed(worker_seed);np.random.seed(worker_seed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--source_path',required=True)
    p.add_argument('--target_path',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--backbone',default='mobilenet_v2',choices=['mobilenet_v2','resnet18','resnet50'])
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch_size',type=int,default=64)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--ssl_weight',type=float,default=.1)
    p.add_argument('--source_weight',type=float,default=1.)
    p.add_argument('--projection_dim',type=int,default=512)
    p.add_argument('--seed',type=int,default=1314)
    p.add_argument('--freeze_backbone_bn',action='store_true',
                   help='freeze feature-extractor BN running statistics; affine gradients remain enabled')
    p.add_argument('--grad_log_interval',type=int,default=0,
                   help='sample weighted loss gradients on shared feature parameters every N steps; 0 disables')
    a=p.parse_args()
    if a.grad_log_interval<0:p.error('grad_log_interval must be nonnegative')
    if a.epochs<1 or a.batch_size<2 or a.workers<0 or a.projection_dim<2:p.error('invalid epochs/batch/workers/dimension')
    if not all(np.isfinite(v) for v in [a.lr,a.ssl_weight,a.source_weight]) or a.lr<=0 or a.ssl_weight<0 or a.source_weight<=0:p.error('invalid learning rate/loss weights')
    import torch
    from torch import nn
    from torchvision import transforms
    import Networks
    from dataset import RafDataSet,FER
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()):p.error('output_dir must be new/empty')
    out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    # Mild geometry preserves facial content, no aggressive crop/color ablation.
    train_transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((256,256)),
        transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(20)],p=.5),transforms.ToTensor(),norm])
    val_transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((256,256)),
        transforms.CenterCrop(224),transforms.ToTensor(),norm])
    source=RafDataSet(a.source_path,phase='train',transform=train_transform,basic_aug=False)
    target=UnlabeledTwoViews(a.target_path,train_transform)
    validation=FER(a.target_path,phase='val',transform=val_transform,basic_aug=False)
    assert_disjoint_paths(target.paths,validation.file_paths)
    if min(len(source),len(target))<a.batch_size:p.error('datasets must contain at least one full batch')
    def loader(dataset,shuffle,drop_last,seed_offset):
        return torch.utils.data.DataLoader(dataset,batch_size=a.batch_size,shuffle=shuffle,
            drop_last=drop_last,num_workers=a.workers,worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(a.seed+seed_offset))
    sl=loader(source,True,True,1);tl=loader(target,True,True,2);vl=loader(validation,False,False,3)
    model=Networks.Model(backbone=a.backbone,num_classes=7,pretrained=False).to(a.device)
    try:ckpt=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    except TypeError:ckpt=torch.load(a.checkpoint,map_location=a.device)
    model.load_state_dict(ckpt['model'],strict=True)
    projector=nn.Sequential(nn.Linear(model.fc.in_features,512),nn.BatchNorm1d(512),
                            nn.ReLU(),nn.Linear(512,a.projection_dim)).to(a.device)
    optimizer=torch.optim.Adam(list(model.parameters())+list(projector.parameters()),lr=a.lr,weight_decay=1e-4)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))

    def evaluate(epoch):
        model.eval();prob=[];truth=[];evaluation_features=[]
        with torch.no_grad():
            for images,labels in vl:
                logits,feature=model(images.to(a.device),None,None,mode='test',task='target')
                evaluation_features.append(feature.numpy())
                prob.append(torch.softmax(logits,1).cpu().numpy());truth.extend(labels.tolist())
        prob=np.concatenate(prob);truth=np.asarray(truth)
        topk,_,_=summarize(prob,truth);metrics=classification_metrics(prob.argmax(1),truth)
        features=np.concatenate(evaluation_features)
        z=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)
        n=len(z);pair_cosine=float((np.square(z.sum(0)).sum()-np.square(z).sum())/(n*(n-1))) if n>1 else None
        encoder_diagnostics=dict(eval_std_mean=float(features.std(0).mean()),
            normalized_eval_std_mean=float(z.std(0).mean()),mean_pairwise_cosine=pair_cosine)
        report=dict(epoch=epoch,split='FER validation only',metrics=metrics,topk=topk,encoder_diagnostics=encoder_diagnostics)
        print('Validation encoder geometry:',encoder_diagnostics)
        (out/('validation_epoch_%02d.json'%epoch)).write_text(json.dumps(report,indent=2,allow_nan=False))
        fear=metrics['per_class'][1]
        print('Validation epoch=%d Acc=%.2f UAR=%.2f MacroF1=%.2f FearRecall=%.2f FearPrecision=%s FearTop2=%.2f FearTop3=%.2f' %
            (epoch,100*metrics['accuracy'],100*metrics['uar'],100*metrics['macro_f1'],100*fear['recall'],
             fear['precision'],100*topk['per_class'][1]['top2']['coverage'],100*topk['per_class'][1]['top3']['coverage']))
        return metrics

    def save(path,epoch,metrics):
        torch.save(dict(model=model.state_dict(),projector=projector.state_dict(),
            optimizer=optimizer.state_dict(),epoch=epoch,args=vars(a),validation=metrics),path)

    print('Target train images:',len(target),'NO target labels parsed/returned; no pseudo-label/EMA/MMD/prototype loss')
    print('Loss = %.3f * source_CE + %.3f * (25*invariance + 25*variance + covariance)' % (a.source_weight,a.ssl_weight))
    print('Freeze backbone BN statistics:',a.freeze_backbone_bn,'Gradient interval:',a.grad_log_interval)
    best=evaluate(-1);best_epoch=-1;save(out/'best.pth',-1,best)
    global_step=0
    with (out/'training.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['epoch','total','source_ce','vicreg','invariance','variance','covariance','projected_std_mean','projected_fraction_std_below1','encoder_std_mean'])
        for epoch in range(a.epochs):
            model.train();projector.train()
            if a.freeze_backbone_bn:
                print('Frozen feature BN modules:',freeze_backbone_bn_statistics(model.feature))
            source_iter=iter(sl);sums=np.zeros(9);batches=0
            for view1,view2 in tl:
                try:images,labels=next(source_iter)
                except StopIteration:source_iter=iter(sl);images,labels=next(source_iter)
                images=images.to(a.device);labels=labels.to(a.device);view1=view1.to(a.device);view2=view2.to(a.device)
                optimizer.zero_grad()
                ns=images.shape[0];nt=view1.shape[0]
                features=model.feature(torch.cat([images,view1,view2],0))
                # Classifier BN is updated by source only; backbone BN sees unlabeled mixed images.
                logits=model.bn(model.fc(features[:ns]))
                ce=nn.functional.cross_entropy(logits,labels)
                f1=features[ns:ns+nt];f2=features[ns+nt:]
                z=projector(torch.cat([f1,f2],0));ssl,parts=vicreg_loss(z[:nt],z[nt:])
                total=a.source_weight*ce+a.ssl_weight*ssl
                if not torch.isfinite(total):raise FloatingPointError('nonfinite adaptation loss')
                if a.grad_log_interval and global_step % a.grad_log_interval == 0:
                    gradient=shared_gradient_metrics(a.source_weight*ce,a.ssl_weight*ssl,model.feature.parameters())
                    print('Gradient step=%d epoch=%d %s' % (global_step,epoch,gradient),flush=True)
                    gradient_path=out/'gradient_diagnostics.csv'
                    fresh=not gradient_path.exists()
                    with gradient_path.open('a',newline='') as gf:
                        gw=csv.DictWriter(gf,fieldnames=['step','epoch']+list(gradient))
                        if fresh:gw.writeheader()
                        gw.writerow(dict(step=global_step,epoch=epoch,**gradient))
                total.backward();optimizer.step();global_step+=1
                encoder_std=(f1.detach().std(0,unbiased=False).mean()+f2.detach().std(0,unbiased=False).mean())/2
                values=[total,ce,ssl,parts['invariance'],parts['variance'],parts['covariance'],
                        parts['projected_std_mean'],parts['projected_fraction_std_below1'],encoder_std]
                sums+=np.array([v.detach().item() for v in values]);batches+=1
            averages=sums/batches
            writer.writerow([epoch]+averages.tolist());f.flush()
            print('Epoch %d total/CE/SSL/inv/var/cov/projstd/stdBelow1/encoderstd: %s' % (epoch,np.round(averages,5)))
            metrics=evaluate(epoch);save(out/'last.pth',epoch,metrics)
            if metrics['macro_f1']>best['macro_f1']:
                best=metrics;best_epoch=epoch;save(out/'best.pth',epoch,metrics)
            print('Best validation MacroF1 %.4f epoch %d (initial baseline epoch=-1)'%(best['macro_f1'],best_epoch))
    print('Finished. Best checkpoint:',out/'best.pth','selected epoch:',best_epoch,'No target test evaluation performed.')

if __name__=='__main__':main()
