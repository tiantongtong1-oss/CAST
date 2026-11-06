"""Equal-weight source true-label and target teacher-probability kNN fusion."""
import torch
from global_medoid_radius import exact_medoid_index
from torch import nn
import torch.nn.functional as F


class FusedKNNBank(nn.Module):
    def __init__(self, feature_dim, num_classes=7, support_threshold=0.7, k=11):
        super().__init__()
        if not 0 <= support_threshold <= 1 or k < 1:
            raise ValueError('support threshold must be in [0,1] and k positive')
        self.register_buffer('class_medoids', torch.zeros(num_classes, feature_dim))
        self.register_buffer('class_center_ready', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('class_counts', torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer('global_radius', torch.zeros(()))
        self.register_buffer('radius_ready', torch.tensor(False))
        self.num_classes = num_classes
        self.k = k
        self.support_threshold = float(support_threshold)
        self.register_buffer('source_features', torch.empty(0, feature_dim), persistent=False)
        self.register_buffer('source_labels', torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer('memory_features', torch.empty(0, feature_dim), persistent=False)
        self.register_buffer('memory_probs', torch.empty(0, num_classes), persistent=False)
        self.register_buffer('memory_ids', torch.empty(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def set_source_memory(self, features, labels):
        features = features.detach().to(self.source_features)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        valid &= (labels >= 0) & (labels < self.num_classes)
        self.source_features = F.normalize(features[valid], dim=1).contiguous()
        self.source_labels = labels[valid].clone()
        self.radius_ready.fill_(False)
        self.global_radius.zero_()
        self.class_medoids.zero_()
        self.class_center_ready.zero_()
        self.class_counts.zero_()
        distance_parts, weight_parts = [], []
        for c in range(self.num_classes):
            z = self.source_features[self.source_labels.eq(c)]
            self.class_counts[c] = z.size(0)
            if z.size(0) < 2:
                continue
            center = z[exact_medoid_index(z)]
            self.class_medoids[c].copy_(center)
            self.class_center_ready[c] = True
            distances = (z.double() - center.double()).norm(dim=1)
            distance_parts.append(distances)
            # Equal total weight per ready class, irrespective of sample count.
            weight_parts.append(torch.full_like(distances, 1.0 / z.size(0)))
        if distance_parts:
            distances = torch.cat(distance_parts)
            weights = torch.cat(weight_parts)
            sorted_distances, order = distances.sort()
            cumulative = weights[order].cumsum(dim=0)
            cumulative = cumulative / cumulative[-1]
            index = torch.searchsorted(cumulative, cumulative.new_tensor(.90))
            index = index.clamp_max(sorted_distances.numel() - 1)
            # Weighted empirical inverse CDF; no interpolation.
            self.global_radius.copy_(sorted_distances[index])
            self.radius_ready.fill_(True)

    @torch.no_grad()
    def set_target_memory(self, features, probabilities, sample_ids):
        features = features.detach().to(self.memory_features)
        probabilities = probabilities.detach().to(self.memory_probs)
        sample_ids = sample_ids.detach().to(device=features.device, dtype=torch.long)
        if probabilities.shape != (features.size(0), self.num_classes) or sample_ids.shape != features.shape[:1]:
            raise ValueError('target memory shape mismatch')
        if sample_ids.unique().numel() != sample_ids.numel():
            raise ValueError('target sample IDs must be unique')
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        valid &= torch.isfinite(probabilities).all(dim=1)
        valid &= (probabilities >= 0).all(dim=1) & (probabilities.sum(dim=1) > 0)
        self.memory_features = F.normalize(features[valid], dim=1).contiguous()
        probs = probabilities[valid]
        self.memory_probs = probs / probs.sum(dim=1, keepdim=True)
        self.memory_ids = sample_ids[valid].clone()

    @torch.no_grad()
    def radius_accepts(self, features, labels):
        features = features.detach().to(self.class_medoids)
        labels = labels.to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        valid &= (labels >= 0) & (labels < self.num_classes)
        safe = labels.clamp(0, self.num_classes - 1)
        z = F.normalize(torch.nan_to_num(features), dim=1)
        distances = (z.double() - self.class_medoids[safe].double()).norm(dim=1)
        return (valid & self.radius_ready & self.class_center_ready[safe]
                & (distances < self.global_radius.double()))

    @torch.no_grad()
    def select(self, features, labels, confidence_mask, agreement_mask, sample_ids):
        features = features.detach().to(self.memory_features)
        labels = labels.to(device=features.device, dtype=torch.long)
        sample_ids = sample_ids.to(device=features.device, dtype=torch.long)
        passed = torch.zeros_like(labels, dtype=torch.bool)
        checked = torch.zeros_like(passed)
        support = features.new_zeros(labels.shape)
        source_support = torch.zeros_like(support)
        target_support = torch.zeros_like(support)
        disagreement = torch.zeros_like(passed)
        if self.source_labels.numel() < self.k or self.memory_ids.numel() < self.k:
            return passed, checked, support, source_support, target_support, disagreement
        eligible = confidence_mask.bool() & agreement_mask.bool()
        eligible &= torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        eligible &= (labels >= 0) & (labels < self.num_classes)
        queries = F.normalize(torch.nan_to_num(features), dim=1)
        for rows in eligible.nonzero(as_tuple=False).flatten().split(128):
            ds = torch.cdist(queries[rows].double(), self.source_features.double())
            si = ds.topk(self.k, dim=1, largest=False).indices
            ps = F.one_hot(self.source_labels[si], self.num_classes).float().mean(dim=1)
            dt = torch.cdist(queries[rows].double(), self.memory_features.double())
            dt.masked_fill_(sample_ids[rows, None].eq(self.memory_ids[None, :]), float('inf'))
            values, ti = dt.topk(self.k, dim=1, largest=False)
            ready = torch.isfinite(values).all(dim=1)
            pt = self.memory_probs[ti].mean(dim=1)
            mixed = 0.5 * ps + 0.5 * pt
            assigned = labels[rows, None]
            s = mixed.gather(1, assigned).squeeze(1)
            checked[rows] = ready
            support[rows] = s
            source_support[rows] = ps.gather(1, assigned).squeeze(1)
            target_support[rows] = pt.gather(1, assigned).squeeze(1)
            disagreement[rows] = ready & ps.argmax(dim=1).ne(pt.argmax(dim=1))
            passed[rows] = ready & mixed.argmax(dim=1).eq(labels[rows]) & (s >= self.support_threshold)
        return passed, checked, support, source_support, target_support, disagreement
