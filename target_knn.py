"""Target-only raw Euclidean kNN, refreshed once per epoch; no radius gate."""
import torch
from torch import nn


class TargetKNNBank(nn.Module):
    def __init__(self, feature_dim, num_classes=7, k=11, min_neighbors=3):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.min_neighbors = min_neighbors
        self.register_buffer('memory_features', torch.empty(0, feature_dim), persistent=False)
        self.register_buffer('memory_labels', torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer('memory_ids', torch.empty(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def set_target_memory(self, features, predictions, sample_ids):
        features = features.detach().to(self.memory_features)
        predictions = predictions.detach().to(device=features.device, dtype=torch.long)
        sample_ids = sample_ids.detach().to(device=features.device, dtype=torch.long)
        if features.ndim != 2 or predictions.shape != features.shape[:1] or sample_ids.shape != predictions.shape:
            raise ValueError('expected features [N,D], predictions [N], IDs [N]')
        if sample_ids.unique().numel() != sample_ids.numel():
            raise ValueError('target sample IDs must be unique')
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        valid &= (predictions >= 0) & (predictions < self.num_classes)
        self.memory_features = features[valid].contiguous()
        self.memory_labels = predictions[valid].clone()
        self.memory_ids = sample_ids[valid].clone()

    @torch.no_grad()
    def select(self, features, labels, confidence_mask, agreement_mask, sample_ids):
        features = features.detach().to(self.memory_features)
        labels = labels.to(device=features.device, dtype=torch.long)
        sample_ids = sample_ids.to(device=features.device, dtype=torch.long)
        passed = torch.zeros_like(labels, dtype=torch.bool)
        checked = torch.zeros_like(passed)
        eligible = confidence_mask.bool() & agreement_mask.bool()
        eligible &= torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        eligible &= (labels >= 0) & (labels < self.num_classes)
        take = min(self.k, self.memory_ids.numel())
        if take < self.min_neighbors:
            return passed, checked
        for rows in eligible.nonzero(as_tuple=False).flatten().split(128):
            distances = torch.cdist(features[rows].double(), self.memory_features.double())
            # Exclude by ID, even if the stored feature differs after teacher updates.
            distances.masked_fill_(sample_ids[rows, None].eq(self.memory_ids[None, :]), float('inf'))
            values, indices = distances.topk(take, dim=1, largest=False)
            finite = torch.isfinite(values)
            counts = finite.sum(dim=1)
            votes = (self.memory_labels[indices].eq(labels[rows, None]) & finite).sum(dim=1)
            checked[rows] = counts >= self.min_neighbors
            passed[rows] = checked[rows] & (votes > counts / 2)
        return passed, checked
