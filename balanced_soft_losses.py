"""Equal-active-class alignment and detached low-weight rescue supervision."""
import math


def equal_class_alignment(source_parts,target_parts,mmd_fn):
    import torch
    if len(source_parts)!=len(target_parts) or not source_parts:raise ValueError('class lists must match')
    losses=[];active=[]
    for c,(source,target) in enumerate(zip(source_parts,target_parts)):
        if source.shape[0]>=2 and target.shape[0]>=2:
            losses.append(mmd_fn(source,target));active.append(c)
    if not losses:
        zero=sum((x.sum()*0 for x in source_parts+target_parts))
        return zero,active
    return torch.stack(losses).mean(),active


def classification_with_soft_rescue(logits,source_labels,pseudo_labels,confidence,
                                    stable_rescue,teacher_probs,rescue_weight=.25):
    import torch
    import torch.nn.functional as F
    if not math.isfinite(rescue_weight) or not 0<=rescue_weight<=1:raise ValueError('rescue weight must be in [0,1]')
    ns=source_labels.numel();nt=pseudo_labels.numel()
    if logits.shape[0]!=ns+nt or teacher_probs.shape!=logits[ns:].shape:raise ValueError('invalid logits/probability shapes')
    confidence=confidence.bool();stable_rescue=stable_rescue.bool() & ~confidence
    if confidence.shape!=(nt,) or stable_rescue.shape!=(nt,):raise ValueError('invalid masks')
    source_sum=F.cross_entropy(logits[:ns],source_labels,reduction='sum')
    hard_per=F.cross_entropy(logits[ns:],pseudo_labels,reduction='none')
    hard_sum=(hard_per*confidence).sum()
    probabilities=teacher_probs.detach()
    soft_per=-(probabilities*F.log_softmax(logits[ns:],dim=1)).sum(1)
    soft_sum=(soft_per*stable_rescue).sum()
    # Denominator counts observations, NOT weighted mass: weight remains genuine
    # attenuation even if every accepted target happens to be rescued.
    denominator=(ns+confidence.sum()+stable_rescue.sum()).clamp_min(1)
    loss=(source_sum+hard_sum+rescue_weight*soft_sum)/denominator
    return loss,dict(source_ce_mean=source_sum/max(ns,1),
        hard_target_mean=hard_sum/confidence.sum().clamp_min(1),
        soft_rescue_mean=soft_sum/stable_rescue.sum().clamp_min(1))
