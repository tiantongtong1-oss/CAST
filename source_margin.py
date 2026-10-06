"""Experiment B: one detached source prototype per true class."""
import torch
import torch.nn.functional as F


def prototype_margin_loss(features, labels, prototypes, margin=0.1):
    """Mean hinge between true-class cosine and hardest other-class cosine."""
    if prototypes.shape[0] < 2:
        raise ValueError('at least two classes are required')
    scores = F.normalize(features, dim=1) @ F.normalize(prototypes.detach(), dim=1).T
    positive = scores.gather(1, labels[:, None]).squeeze(1)
    competitors = scores.masked_fill(
        F.one_hot(labels, prototypes.shape[0]).bool(), float('-inf'))
    negative = competitors.max(dim=1).values
    return F.relu(margin + negative - positive).mean()
