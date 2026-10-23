"""Post-inference label-based metrics; never used by model or training."""
import numpy as np

NAMES=['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']


def fraction(a,b):
    return float(a/b) if b else None


def distribution(values):
    if not len(values): return None
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),
                p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))


def summarize(probabilities,truth):
    p=np.asarray(probabilities,dtype=np.float64); truth=np.asarray(truth,dtype=np.int64)
    if p.ndim!=2 or p.shape[1]!=7 or truth.shape!=(len(p),) or not len(p):
        raise ValueError('require nonempty N x 7 probabilities and N labels')
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('invalid probabilities')
    if ((truth<0)|(truth>=7)).any():raise ValueError('invalid labels')
    order=np.argsort(-p,axis=1,kind='stable')
    pred=order[:,0]
    ranks=np.argsort(order,axis=1,kind='stable')+1
    cm=np.bincount(truth*7+pred,minlength=49).reshape(7,7)
    classes=[]
    for c,name in enumerate(NAMES):
        true=truth==c
        result=dict(class_name=name,true_support=int(true.sum()))
        for k in [1,2,3]:
            candidate=ranks[:,c]<=k
            correct=int((true&candidate).sum()); count=int(candidate.sum())
            result['top%d'%k]=dict(candidate_count=count,correct=correct,
                coverage=fraction(correct,int(true.sum())),candidate_precision=fraction(correct,count))
        classes.append(result)
    fear=truth==1; gap=p.max(1)-p[:,1]
    conditional=[]
    for c in [4,6]:
        group=fear&(pred==c); count=int(group.sum())
        conditional.append(dict(predicted_as=NAMES[c],true_fear_count=count,
            fear_rank2_count=int((group&(ranks[:,1]==2)).sum()),
            fear_rank2_fraction=fraction(int((group&(ranks[:,1]==2)).sum()),count),
            fear_top3_fraction=fraction(int((group&(ranks[:,1]<=3)).sum()),count),
            top1_minus_fear_probability=distribution(gap[group])))
    return dict(samples=len(p),accuracy=float((pred==truth).mean()),
        class_order=NAMES,per_class=classes,fear_confusion_conditionals=conditional,
        fear_probability_gap=dict(all_true_fear=distribution(gap[fear]),
            missed_true_fear=distribution(gap[fear&(pred!=1)])),
        confusion_rows_true_columns_predicted=cm.tolist()),order,ranks
