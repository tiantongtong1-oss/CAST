"""Source-only prototypes and mean of the farthest-k Euclidean distances.

Historical Gaussian experiments live in global_gaussian_knn_reliability.py;
this bank does not calculate variances or Mahalanobis distances.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class SourcePrototypeRadiusBank(nn.Module):
    def __init__(self, num_classes, feature_dim, source_radius_k=11, eps=1e-8):
        super().__init__()
        for name, value in (("num_classes", num_classes), ("feature_dim", feature_dim),
                            ("source_radius_k", source_radius_k)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(name + " must be a positive integer")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.num_classes, self.feature_dim = int(num_classes), int(feature_dim)
        self.source_radius_k, self.eps = int(source_radius_k), float(eps)
        self.register_buffer("class_means", torch.zeros(num_classes, feature_dim))
        self.register_buffer("class_radii", torch.zeros(num_classes))
        self.register_buffer("distribution_initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("class_source_counts", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("source_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("refresh_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("_source_sum", torch.zeros(num_classes, feature_dim, dtype=torch.float64), persistent=False)
        self.register_buffer("_source_count", torch.zeros(num_classes, dtype=torch.long), persistent=False)
        # CPU snapshots avoid retaining the whole source dataset on GPU. Distances
        # must be computed after the final class means are known (not batch means).
        self._source_batches = []

    @torch.no_grad()
    def begin_source_refresh(self):
        self._source_sum.zero_()
        self._source_count.zero_()
        self._source_batches.clear()

    @torch.no_grad()
    def accumulate_source(self, features, labels):
        features = features.detach().to(self.class_means)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        if features.ndim != 2 or features.shape[1] != self.feature_dim or labels.shape != (features.shape[0],):
            raise ValueError("invalid source feature/label shapes")
        valid = torch.isfinite(features).all(1) & (features.norm(dim=1) > self.eps)
        valid &= (labels >= 0) & (labels < self.num_classes)
        z = F.normalize(features[valid], dim=1)
        labels = labels[valid]
        if not labels.numel():
            return
        self._source_sum.index_add_(0, labels, z.double())
        self._source_count.add_(torch.bincount(labels, minlength=self.num_classes))
        self._source_batches.append((z.cpu(), labels.cpu()))

    @torch.no_grad()
    def finalize_source(self):
        self.class_source_counts.copy_(self._source_count)
        self.source_count.copy_(self._source_count.sum())
        self.class_means.copy_((self._source_sum / self._source_count.clamp_min(1)[:, None]).to(self.class_means))
        self.class_radii.zero_()
        self.distribution_initialized.copy_(self._source_count > 0)
        top_distances = [self.class_radii.new_empty(0) for _ in range(self.num_classes)]
        for features, labels in self._source_batches:
            features, labels = features.to(self.class_means), labels.to(self.class_means.device)
            distances = (features - self.class_means[labels]).norm(dim=1)
            for c in labels.unique().tolist():
                values = torch.cat((top_distances[c], distances[labels == c]))
                top_distances[c] = values.topk(min(self.source_radius_k, values.numel()), largest=True).values
        for c, values in enumerate(top_distances):
            if values.numel():
                self.class_radii[c] = values.mean()
        self._source_batches.clear()
        self.refresh_count.add_(1)

    @torch.no_grad()
    def distribution_distance(self, features, labels):
        features = features.detach().to(self.class_means)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        if features.shape[-1] != self.feature_dim or labels.shape != features.shape[:-1]:
            raise ValueError("invalid distribution feature/label shapes")
        safe = labels.clamp(0, self.num_classes - 1)
        valid = (labels >= 0) & (labels < self.num_classes) & self.distribution_initialized[safe]
        valid &= torch.isfinite(features).all(-1) & (features.norm(dim=-1) > self.eps)
        z = F.normalize(torch.nan_to_num(features), dim=-1)
        distance = (z - self.class_means[safe]).norm(dim=-1)
        return distance.masked_fill(~valid, float("inf"))

    @torch.no_grad()
    def in_distribution(self, features, labels):
        labels = labels.to(device=self.class_means.device, dtype=torch.long)
        # Strict inequality implements the requested rule, including radius=0.
        return self.distribution_distance(features, labels) < self.class_radii[labels.clamp(0, self.num_classes - 1)]

