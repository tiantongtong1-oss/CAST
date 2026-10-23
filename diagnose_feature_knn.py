"""Validation-only comparison: classifier vs global and per-class source KNN."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from feature_knn_diagnostics import predict_from_source,compare_predictions
from topk_diagnostics import NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--source_path',required=True)
    p.add_argument('--target_path',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--backbone',default='mobilenet_v2',choices=['mobilenet_v2','resnet18','resnet50'])
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--batch_size',type=int,default=128)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--k',type=int,default=11)
    p.add_argument('--query_chunk',type=int,default=64)
    a=p.parse_args()
    if min(a.batch_size,a.k,a.query_chunk)<1 or a.workers<0:p.error('invalid batch/K/chunk/workers')
    import torch
    from torchvision import transforms
    import Networks
    from dataset import RafDataSet,FER
    from prototype_utils import FeatureHook
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; use --device cpu')
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()):p.error('output_dir must be new or empty')
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(1314);np.random.seed(1314)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if a.device=='cuda' and hasattr(torch.backends.cuda.matmul,'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32=False
    transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((256,256)),
        transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    source=RafDataSet(a.source_path,phase='train',transform=transform,basic_aug=False)
    target=FER(a.target_path,phase='val',transform=transform,basic_aug=False,return_index=True)
    model=Networks.Model(backbone=a.backbone,num_classes=7,pretrained=False).to(a.device)
    try:checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    except TypeError:checkpoint=torch.load(a.checkpoint,map_location=a.device)
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    hook=FeatureHook(model.feature)
    sx=[];sy=[];tx=[];prob=[];ids=[]
    try:
        with torch.no_grad():
            for images,labels in torch.utils.data.DataLoader(source,batch_size=a.batch_size,shuffle=False,num_workers=a.workers):
                model(images.to(a.device),None,None,mode='test',task='source')
                sx.append(hook.output.detach().cpu().numpy());sy.append(labels.numpy())
            for images,_unused_truth,batch_ids in torch.utils.data.DataLoader(target,batch_size=a.batch_size,shuffle=False,num_workers=a.workers):
                logits,_=model(images.to(a.device),None,None,mode='test',task='target')
                tx.append(hook.output.detach().cpu().numpy())
                prob.append(torch.softmax(logits,dim=1).cpu().numpy());ids.extend(batch_ids.tolist())
    finally:hook.close()
    assert ids==list(range(len(target)))
    sx=np.concatenate(sx);sy=np.concatenate(sy);tx=np.concatenate(tx);prob=np.concatenate(prob)
    print('Source refs:',len(sx),'Class counts:',np.bincount(sy,minlength=7).tolist())
    predictions=predict_from_source(sx,sy,tx,a.k,a.query_chunk,a.device)
    head=prob.argmax(1)
    # Save all predictions/evidence before joining target truth for evaluation.
    with (out/'raw_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','image_path','head','knn','classwise']+
            ['p_'+n for n in NAMES]+['votes_'+n for n in NAMES]+['distance_'+n for n in NAMES])
        for i in ids:w.writerow([i,target.file_paths[i],NAMES[head[i]],NAMES[predictions['knn'][i]],
            NAMES[predictions['classwise'][i]]]+prob[i].tolist()+predictions['votes'][i].tolist()+predictions['classwise_mean_distance'][i].tolist())
    truth=np.asarray(target.label)
    result=compare_predictions(head,predictions['knn'],predictions['classwise'],truth)
    summary=dict(checkpoint=str(Path(a.checkpoint).resolve()),split='FER validation only',
        source_split='RAF training references',k=a.k,class_order=NAMES,
        feature='model.feature; L2 normalization; Euclidean distance',methods=result)
    (out/'feature_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (out/'confusions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','true_class']+NAMES)
        for name,m in result.items():
            for c,row in enumerate(m['confusion_rows_true_columns_predicted']):w.writerow([name,NAMES[c]]+row)
    with (out/'per_image_evaluation.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','true_class','head','knn','classwise'])
        for i in ids:w.writerow([i,NAMES[truth[i]],NAMES[head[i]],NAMES[predictions['knn'][i]],NAMES[predictions['classwise'][i]]])
    for name,m in result.items():
        print(name,'Accuracy %.2f UAR %.2f MacroF1 %.2f' % (100*m['accuracy'],100*m['uar'],100*m['macro_f1']))
        print('Fear:',m['per_class'][1]);print('Fear vs head:',m['fear_vs_head'])
    print('Saved:',out.resolve())

if __name__=='__main__':main()
