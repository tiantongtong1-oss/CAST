"""FER validation or RAF official test comparison against RAF training references."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from feature_knn_diagnostics import predict_from_source,compare_predictions
from topk_diagnostics import NAMES
from input_image_audit import audit_images,save_preview
from resolution_diagnostic import LowResolutionBottleneck,save_resolution_preview
from diagnostic_data_audit import assert_disjoint_paths,audit_raf_labels,EXPECTED_FER_TO_CAST


class IndexedQuery:
    def __init__(self,dataset): self.dataset=dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self,index):
        image,label=self.dataset[index]
        return image,label,index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--source_path',required=True)
    p.add_argument('--target_path',help='required only for fer_val queries')
    p.add_argument('--query_dataset',choices=['fer_val','raf_test'],default='fer_val')
    p.add_argument('--output_dir',required=True)
    p.add_argument('--backbone',default='mobilenet_v2',choices=['mobilenet_v2','resnet18','resnet50'])
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--batch_size',type=int,default=128)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--k',type=int,default=11)
    p.add_argument('--query_chunk',type=int,default=64)
    p.add_argument('--query_lowres_size',type=int,default=0,
                   help='0 keeps native query input; 48 inserts bilinear 48x48 bottleneck before Resize256')
    p.add_argument('--reference_lowres_size',type=int,default=0,
                   help='reference bottleneck; default 0 keeps original RAF training reference features')
    p.add_argument('--color_mode',choices=['rgb','gray'],default='rgb',
                   help='query color; gray converts to grayscale and replicates three channels before normalization')
    p.add_argument('--reference_color_mode',choices=['same','rgb','gray'],default='same',
                   help='reference color; same applies query color to both domains')
    p.add_argument('--audit_images',type=int,default=32,
                   help='first N images per domain audited, 0 disables; previews first six')
    a=p.parse_args()
    if a.query_lowres_size<0 or a.reference_lowres_size<0:p.error('lowres sizes must be nonnegative')
    if not 0 <= a.audit_images <= 256:p.error('audit_images must be in [0,256]')
    if min(a.batch_size,a.k,a.query_chunk)<1 or a.workers<0:p.error('invalid batch/K/chunk/workers')
    if a.query_dataset=='fer_val' and not a.target_path:p.error('--target_path required for fer_val')
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
    def make_transform(mode,lowres_size):
        steps=[transforms.ToPILImage(),LowResolutionBottleneck(lowres_size),
               transforms.Resize((256,256)),transforms.CenterCrop(224)]
        if mode=='gray':steps.append(transforms.Grayscale(num_output_channels=3))
        steps += [transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        return transforms.Compose(steps)
    reference_mode=a.color_mode if a.reference_color_mode=='same' else a.reference_color_mode
    transform=make_transform(a.color_mode,a.query_lowres_size)
    reference_transform=make_transform(reference_mode,a.reference_lowres_size)
    source=RafDataSet(a.source_path,phase='train',transform=reference_transform,basic_aug=False)
    audit=dict(reference=audit_raf_labels(a.source_path,source.file_paths,source.label,'train'))
    if dict(FER.FER_TO_CAST)!=EXPECTED_FER_TO_CAST:raise ValueError('unexpected FER label mapping')
    if a.query_dataset=='raf_test':
        target=RafDataSet(a.source_path,phase='test',transform=transform,basic_aug=False)
        query_loader_dataset=IndexedQuery(target)
        audit['query']=audit_raf_labels(a.source_path,target.file_paths,target.label,'test')
    else:
        target=FER(a.target_path,phase='val',transform=transform,basic_aug=False,return_index=True)
        query_loader_dataset=target
    audit['split_check']=assert_disjoint_paths(source.file_paths,target.file_paths)
    # Verify the current loaders retain the explicit BGR -> RGB conversion.
    import inspect
    for cls in [RafDataSet,FER]:
        code=inspect.getsource(cls.__getitem__).replace(' ','')
        if 'image=image[:,:,::-1]' not in code:raise ValueError('BGR/RGB conversion changed in '+cls.__name__)
    audit['raf_raw_to_class']={str(i+1):name for i,name in enumerate(NAMES)}
    audit['fer_raw_to_class']={str(i):NAMES[c] for i,c in FER.FER_TO_CAST.items()}
    audit['channels']='Both loader implementations explicitly convert cv2 BGR to RGB'
    audit['transform']='[optional bilinear lowres] Resize256 CenterCrop224 [optional Grayscale3] ToTensor ImageNet normalization; basic_aug=False'
    audit['resolution_modes']=dict(query_lowres_size=a.query_lowres_size,
        reference_lowres_size=a.reference_lowres_size,zero_means='native image',
        bottleneck_interpolation='PIL bilinear',note='Reference and query bottlenecks are independent')
    audit['color_modes']=dict(query=a.color_mode,reference=reference_mode,
        note='Three equal grayscale channels BEFORE normalization; per-channel ImageNet normalization remains unchanged')
    if a.audit_images:
        audit['image_samples']=dict(reference=audit_images(source.file_paths,a.audit_images),
                                    query=audit_images(target.file_paths,a.audit_images))
        save_preview(source.file_paths,target.file_paths,out/'input_preview.jpg')
        if a.query_lowres_size:
            save_resolution_preview(target.file_paths,out/'resolution_preview.jpg',a.query_lowres_size)
        (out/'input_audit.json').write_text(json.dumps(audit,indent=2))
    print('Dataset audit:',json.dumps(audit,ensure_ascii=False))
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
            for images,_unused_truth,batch_ids in torch.utils.data.DataLoader(query_loader_dataset,batch_size=a.batch_size,shuffle=False,num_workers=a.workers):
                logits,_=model(images.to(a.device),None,None,mode='test',task=('source' if a.query_dataset=='raf_test' else 'target'))
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
    summary=dict(checkpoint=str(Path(a.checkpoint).resolve()),split=('RAF official test only' if a.query_dataset=='raf_test' else 'FER validation only'),
        dataset_audit=audit,
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
