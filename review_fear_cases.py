"""One-pass visual diagnosis of validation Fear errors and source neighbors."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from topk_diagnostics import NAMES
from fear_review_utils import select_review_groups,nearest_source,model_crop_image,render_pages
from diagnostic_data_audit import audit_raf_labels,assert_disjoint_paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--predictions',required=True,help='existing raw_probabilities.csv or raw_predictions.csv with p_Class columns')
    parser.add_argument('--source_path',required=True)
    parser.add_argument('--target_path',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--backbone',default='mobilenet_v2',choices=['mobilenet_v2','resnet18','resnet50'])
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--min_confidence',type=float,default=.8)
    a=parser.parse_args()
    if a.count<1 or a.count>100 or a.batch_size<1 or not 0<=a.min_confidence<=1:parser.error('invalid count/batch/confidence')
    import torch
    from torchvision import transforms
    import Networks
    from dataset import RafDataSet,FER
    from prototype_utils import FeatureHook
    if a.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable; use --device cpu')
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()):parser.error('output_dir must be new/empty')
    out.mkdir(parents=True,exist_ok=True)
    transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((256,256)),
        transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    torch.manual_seed(1314);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    source=RafDataSet(a.source_path,phase='train',transform=transform,basic_aug=False)
    target=FER(a.target_path,phase='val',transform=transform,basic_aug=False)
    audit_raf_labels(a.source_path,source.file_paths,source.label,'train')
    assert_disjoint_paths(source.file_paths,target.file_paths)
    with open(a.predictions,newline='') as f:records=list(csv.DictReader(f))
    if len(records)!=len(target):raise ValueError('prediction file must cover complete FER validation set')
    mapping={int(r['sample_id']):r for r in records}
    if set(mapping)!=set(range(len(target))):raise ValueError('duplicate/missing validation IDs')
    for i,path in enumerate(target.file_paths):
        if Path(mapping[i]['image_path']).resolve()!=Path(path).resolve():raise ValueError('prediction path/validation ID mismatch')
    probability=np.array([[float(mapping[i]['p_'+name]) for name in NAMES] for i in range(len(target))])
    if not np.isfinite(probability).all() or (probability<0).any() or not np.allclose(probability.sum(1),1,atol=1e-5):raise ValueError('invalid saved probabilities')
    # True validation labels are used ONLY to select review examples, not train.
    groups=select_review_groups(probability,np.array(target.label),a.count,a.min_confidence)
    selected=np.unique(np.concatenate(list(groups.values())))
    if not len(selected):raise ValueError('no selected validation cases; inspect saved predictions')
    model=Networks.Model(backbone=a.backbone,num_classes=7,pretrained=False).to(a.device)
    try:checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    except TypeError:checkpoint=torch.load(a.checkpoint,map_location=a.device)
    model.load_state_dict(checkpoint['model'],strict=True);model.eval();hook=FeatureHook(model.feature)
    source_features=[];source_probs=[];query_features=[];crops={}
    fear_ids=np.flatnonzero(np.asarray(source.label)==1)
    source_review=fear_ids[np.linspace(0,len(fear_ids)-1,min(a.count,len(fear_ids)),dtype=int)]
    try:
        with torch.no_grad():
            # Reuse stored validation predictions; re-forward only chosen cases.
            for begin in range(0,len(selected),a.batch_size):
                ids=selected[begin:begin+a.batch_size]
                images=torch.stack([target[int(i)][0] for i in ids])
                logits,_=model(images.to(a.device),None,None,mode='test',task='target')
                fresh=torch.softmax(logits,1).cpu().numpy()
                if not np.allclose(fresh,probability[ids],atol=2e-4,rtol=2e-3):
                    raise ValueError('saved probabilities do not match this checkpoint/default RGB crop; use matching predictions')
                query_features.extend(hook.output.detach().cpu().numpy())
                for i,image in zip(ids,images):crops[int(i)]=model_crop_image(image.numpy())
            # Only one pass through source reference images; no KNN validation sweep.
            for images,_labels in torch.utils.data.DataLoader(source,batch_size=a.batch_size,shuffle=False,num_workers=0):
                logits,_=model(images.to(a.device),None,None,mode='test',task='source')
                source_features.append(hook.output.detach().cpu().numpy())
                source_probs.append(torch.softmax(logits,1).cpu().numpy())
    finally:hook.close()
    sx=np.concatenate(source_features);sp=np.concatenate(source_probs);q=np.asarray(query_features)
    idx,sims=nearest_source(sx,q,3)
    crop_source_ids=set(idx.ravel().tolist())|set(source_review.tolist())
    source_crops={i:model_crop_image(source[i][0].numpy()) for i in crop_source_ids}
    position={int(i):j for j,i in enumerate(selected)}
    manifest=[];output_files=[]
    for group,ids in groups.items():
        rows=[]
        for i in ids:
            i=int(i);j=position[i];pred=int(probability[i].argmax());neighbors=[];neighbor_meta=[]
            for neighbor,similarity in zip(idx[j],sims[j]):
                n=int(neighbor);label=int(source.label[n])
                neighbors.append(dict(crop=source_crops[n],caption='%s cos=%.3f'%(NAMES[label],similarity)))
                neighbor_meta.append(dict(source_id=n,path=source.file_paths[n],true_class=NAMES[label],cosine=float(similarity)))
            rows.append(dict(path=target.file_paths[i],crop=crops[i],neighbors=neighbors,
                caption='id=%d true=Fear pred=%s p=%.3f pFear=%.3f'%(i,NAMES[pred],probability[i,pred],probability[i,1])))
            manifest.append(dict(group=group,sample_id=i,path=target.file_paths[i],true_class='Fear',
                probabilities=probability[i].tolist(),neighbors=neighbor_meta))
        output_files+=render_pages(rows,out,group)
    source_rows=[]
    for i in source_review:
        i=int(i);pred=int(sp[i].argmax())
        source_rows.append(dict(path=source.file_paths[i],crop=source_crops[i],caption=
            'RAF train id=%d true=Fear pred=%s p=%.3f pFear=%.3f'%(i,NAMES[pred],sp[i,pred],sp[i,1])))
    output_files+=render_pages(source_rows,out,'raf_fear_reference')
    np.savez_compressed(out/'features_cache.npz',source_features=sx,source_labels=source.label,
        selected_validation_ids=selected,selected_validation_features=q)
    report=dict(checkpoint=str(Path(a.checkpoint).resolve()),predictions=str(Path(a.predictions).resolve()),
        scope='FER validation review only; no training or label correction',class_order=NAMES,
        min_confidence=a.min_confidence,counts={g:len(ids) for g,ids in groups.items()},
        source_reference_cases=[dict(source_id=int(i),path=source.file_paths[int(i)],true_class='Fear',probabilities=sp[int(i)].tolist()) for i in source_review],
        source_reference_count=len(source_review),source_reference_selection='evenly spaced within existing shuffled RAF Fear list',
        predictions_match_checkpoint=True,pages=output_files,cases=manifest,
        limitations='Hand-picked diagnostic examples, not a representative sample or proof of annotation error; no auto diagnosis')
    (out/'review_manifest.json').write_text(json.dumps(report,indent=2))
    print('Selected groups:',report['counts'],'RAF Fear examples:',len(source_review))
    if len(groups['failed_fear'])<a.count:print('Fewer errors meet min_confidence; no lower-confidence backfill.')
    print('Saved contact sheets and review_manifest.json:',out.resolve())

if __name__=='__main__':main()
