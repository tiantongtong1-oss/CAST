"""Experiment C: closest positive mode versus hardest other-class mode."""
import torch
import torch.nn.functional as F


def multi_prototype_margin_loss(features, labels, prototypes, margin=0.1):
    if prototypes.ndim != 3 or prototypes.shape[0] < 2:
        raise ValueError('prototypes must have shape classes x modes x dimension')
    centers = F.normalize(prototypes.detach(), dim=-1)
    similarities = torch.einsum('bd,ckd->bck', F.normalize(features, dim=1), centers)
    class_scores = similarities.max(dim=2).values
    positive = class_scores.gather(1, labels[:, None]).squeeze(1)
    negative = class_scores.masked_fill(
        F.one_hot(labels, prototypes.shape[0]).bool(), float('-inf')).max(1).values
    return F.relu(margin + negative - positive).mean()
