"""FER validation-only, deterministic inference and post-hoc top-k diagnosis."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from topk_diagnostics import summarize,NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--target_path',required=True)
    parser.add_argument('--backbone',default='mobilenet_v2',choices=['mobilenet_v2','resnet18','resnet50'])
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--output_dir',required=True)
    args=parser.parse_args()
    if args.batch_size<1 or args.workers<0: parser.error('invalid batch size or workers')
    # Deferred imports let --help and numerical tests work without PyTorch.
    import torch
    from torchvision import transforms
    import Networks
    from dataset import FER
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable; use --device cpu')
    outdir=Path(args.output_dir)
    if outdir.exists() and any(outdir.iterdir()):parser.error('output_dir must be empty or new to avoid overwriting results')
    outdir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(1314)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    transform=transforms.Compose([
        transforms.ToPILImage(),transforms.Resize((256,256)),transforms.CenterCrop(224),
        transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    dataset=FER(args.target_path,phase='val',transform=transform,basic_aug=False,return_index=True)
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=False,
                                       num_workers=args.workers)
    # Full checkpoint supplies all parameters; no ImageNet download is needed.
    model=Networks.Model(backbone=args.backbone,num_classes=7,pretrained=False).to(args.device)
    try: checkpoint=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    except TypeError: checkpoint=torch.load(args.checkpoint,map_location=args.device)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    probabilities=[]; ids=[]
    with torch.no_grad():
        for images,_unused_labels,batch_ids in loader:
            logits,_=model(images.to(args.device),None,None,mode='test',task='target')
            probabilities.append(torch.softmax(logits,dim=1).cpu().numpy())
            ids.extend(batch_ids.tolist())
    p=np.concatenate(probabilities)
    assert ids==list(range(len(dataset))), 'unexpected loader order'
    # Save raw predictions BEFORE joining true labels for post-hoc metrics.
    with (outdir/'raw_probabilities.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['sample_id','image_path']+['p_'+n for n in NAMES])
        for i,prob in zip(ids,p):writer.writerow([i,dataset.file_paths[i]]+prob.tolist())
    truth=np.asarray(dataset.label,dtype=np.int64)
    summary,order,ranks=summarize(p,truth)
    summary['checkpoint']=str(Path(args.checkpoint).resolve())
    summary['split']='FER validation only'
    summary['preprocessing']='Resize256 CenterCrop224 ImageNet normalization; deterministic single view'
    (outdir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    with (outdir/'per_image_diagnostics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['sample_id','true_class','top1','top2','top3','fear_rank','top1_minus_fear_probability'])
        for i in ids:writer.writerow([i,NAMES[truth[i]]]+[NAMES[c] for c in order[i,:3]]+[int(ranks[i,1]),float(p[i].max()-p[i,1])])
    with (outdir/'confusion.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['true_class']+NAMES)
        for name,row in zip(NAMES,summary['confusion_rows_true_columns_predicted']):writer.writerow([name]+row)
    print('Validation accuracy: %.2f%%' % (100*summary['accuracy']))
    for item in summary['per_class']:
        print(item['class_name'],json.dumps(item,ensure_ascii=False))
    print('Fear confused with Sad/Neutral:',json.dumps(summary['fear_confusion_conditionals'],ensure_ascii=False,indent=2))
    print('Fear probability gaps:',json.dumps(summary['fear_probability_gap'],indent=2))
    print('Saved diagnostics:',outdir.resolve())

if __name__=='__main__':main()
