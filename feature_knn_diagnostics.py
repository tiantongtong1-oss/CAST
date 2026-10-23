"""Independent source-reference predictions; target labels are not inputs."""
import numpy as np
from topk_diagnostics import NAMES, fraction


def normalize(features):
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all() or (np.linalg.norm(x,axis=1)<=1e-8).any():
        raise ValueError('features must be finite nonzero row vectors')
    return x/np.linalg.norm(x,axis=1,keepdims=True)


def predict_from_source(source_features,source_labels,target_features,k=11,chunk_size=64,device='cpu'):
    source=normalize(source_features); target=normalize(target_features)
    labels=np.asarray(source_labels,dtype=np.int64)
    if labels.shape!=(len(source),) or ((labels<0)|(labels>=7)).any():raise ValueError('invalid source labels')
    if k<1 or chunk_size<1 or len(source)<k:raise ValueError('invalid K/chunk size or insufficient source references')
    indices=[np.flatnonzero(labels==c) for c in range(7)]
    if any(len(i)<k for i in indices):raise ValueError('every source class must contain at least K references')
    gpu_source=None
    if device!='cpu':
        import torch
        gpu_source=torch.as_tensor(source,device=device)
    votes=np.zeros((len(target),7),dtype=np.int64)
    means=np.empty((len(target),7),dtype=np.float32)
    for start in range(0,len(target),chunk_size):
        block=target[start:start+chunk_size]
        if gpu_source is None:sim=block@source.T
        else:
            import torch
            with torch.no_grad():sim=(torch.as_tensor(block,device=device)@gpu_source.T).cpu().numpy()
        distances=np.sqrt(np.clip(2-2*sim,0,4))
        # Stable ordering: source row index breaks equal-distance ties.
        nearest=np.argsort(distances,axis=1,kind='stable')[:,:k]
        for offset,neighbor in enumerate(nearest):
            votes[start+offset]=np.bincount(labels[neighbor],minlength=7)
        for c,idx in enumerate(indices):
            local=distances[:,idx]
            means[start:start+len(block),c]=np.partition(local,k-1,axis=1)[:,:k].mean(1)
    # Equal class votes / equal mean distances break ties by class index.
    return dict(knn=votes.argmax(1),classwise=means.argmin(1),votes=votes,classwise_mean_distance=means)


def classification_metrics(pred,truth):
    pred=np.asarray(pred,dtype=np.int64); truth=np.asarray(truth,dtype=np.int64)
    if pred.shape!=truth.shape or not len(pred):raise ValueError('invalid prediction shapes')
    cm=np.bincount(truth*7+pred,minlength=49).reshape(7,7)
    support=cm.sum(1); count=cm.sum(0); correct=cm.diagonal()
    recall=np.divide(correct,support,out=np.zeros(7,dtype=float),where=support>0)
    f1=np.divide(2*correct,support+count,out=np.zeros(7,dtype=float),where=support+count>0)
    return dict(accuracy=float(correct.sum()/len(pred)),uar=float(recall[support>0].mean()),
        macro_f1=float(f1.mean()),confusion_rows_true_columns_predicted=cm.tolist(),
        per_class=[dict(class_name=NAMES[c],true_support=int(support[c]),predicted_count=int(count[c]),
            correct=int(correct[c]),precision=fraction(int(correct[c]),int(count[c])),
            recall=fraction(int(correct[c]),int(support[c])),f1=float(f1[c])) for c in range(7)])


def compare_predictions(head,knn,classwise,truth):
    truth=np.asarray(truth); head=np.asarray(head)
    result={}
    for name,pred in [('classifier_head',head),('source_knn',knn),('classwise_distance',classwise)]:
        pred=np.asarray(pred); metrics=classification_metrics(pred,truth)
        new=(head!=1)&(pred==1)
        rescued=new&(truth==1); lost=(head==1)&(truth==1)&(pred!=1)
        metrics['fear_vs_head']=dict(new_fear_predictions=int(new.sum()),
            recovered_true_fear=int(rescued.sum()),new_fear_precision=fraction(int(rescued.sum()),int(new.sum())),
            lost_head_correct_fear=int(lost.sum()),net_correct_fear_change=int(rescued.sum()-lost.sum()))
        result[name]=metrics
    return result
