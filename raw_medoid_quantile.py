"""L2-normalized medoid/Q90 radius gate, and L2-normalized source kNN."""
import torch
import torch.nn.functional as F
from global_medoid_radius import GlobalMedoidRadiusBank, exact_medoid_index


class RawMedoidQuantileBank(GlobalMedoidRadiusBank):
    @torch.no_grad()
    def set_source_memory(self, features, labels):
        features = features.detach().to(self.medoid)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        valid &= (labels >= 0) & (labels < self.prototypes.size(0))
        # Keep raw memory for prototype/radius compatibility; kNN normalizes separate copies.
        self.memory_features = features[valid].contiguous()
        self.memory_labels = labels[valid].clone()
        self.prototypes.zero_()
        self.initialized.zero_()
        self.counts.zero_()
        self.ready.zero_()
        self.radii.zero_()
        self._farthest.fill_(-float('inf'))
        for c in range(self.prototypes.size(0)):
            rows = self.memory_labels.eq(c)
            self.counts[c] = rows.sum()
            if rows.any():
                # Separate angular prototypes only for inherited consistency loss;
                # never used as the radius center.
                total = F.normalize(self.memory_features[rows], dim=1).sum(dim=0)
                if total.norm() > 1e-8:
                    self.prototypes[c] = F.normalize(total, dim=0)
                    self.initialized[c] = True
        self.global_ready.fill_(False)
        self.global_radius.zero_()
        self.medoid.zero_()
        self.medoid_memory_index.fill_(-1)
        if self.memory_features.size(0) < 2:
            return
        radius_features = F.normalize(self.memory_features, dim=1)
        index = exact_medoid_index(radius_features)
        self.medoid_memory_index.copy_(index)
        self.medoid.copy_(radius_features[index])
        distances = (radius_features.double() - self.medoid.double()).norm(dim=1)
        # Exact empirical 90th percentile, with linear interpolation.
        self.global_radius.copy_(torch.quantile(distances, 0.90, interpolation='linear'))
        self.global_ready.fill_(True)

    @torch.no_grad()
    def accepts(self, features, labels):
        features = features.detach().to(self.medoid)
        labels = labels.to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        valid &= (labels >= 0) & (labels < self.prototypes.size(0))
        radius_features = F.normalize(torch.nan_to_num(features), dim=1)
        distances = (radius_features.double() - self.medoid.double()).norm(dim=1)
        return valid & self.global_ready & (distances < self.global_radius.double())

    @torch.no_grad()
    def neighbor_accepts(self, features, labels, candidates, k=11, min_neighbors=3):
        features = features.detach().to(self.medoid)
        labels = labels.to(device=features.device, dtype=torch.long)
        passed = torch.zeros_like(labels, dtype=torch.bool)
        checked = torch.zeros_like(passed)
        take = min(k, self.memory_features.size(0))
        if take < min_neighbors:
            return passed, checked
        valid = candidates.bool() & torch.isfinite(features).all(dim=1)
        valid &= features.norm(dim=1) > 1e-8
        valid &= (labels >= 0) & (labels < self.prototypes.size(0))
        queries = F.normalize(torch.nan_to_num(features), dim=1)
        neighbors = F.normalize(self.memory_features, dim=1)
        for rows in valid.nonzero(as_tuple=False).flatten().split(128):
            distances = torch.cdist(queries[rows].double(), neighbors.double())
            indices = distances.topk(take, dim=1, largest=False).indices
            votes = self.memory_labels[indices].eq(labels[rows, None]).sum(dim=1)
            passed[rows] = votes > take / 2
            checked[rows] = True
        return passed, checked


    @torch.no_grad()
    def select(self, features, labels, confidence_mask, agreement_mask, enabled):
        """Require kNN after confidence OR radius; enabled controls radius warmup only.

        Returns final acceptance, accepted radius rescues, kNN passes and checks.
        kNN is mandatory from epoch zero, including for high-confidence samples.
        """
        direct = confidence_mask.bool() & agreement_mask.bool()
        radius = torch.zeros_like(direct)
        if enabled:
            candidates = agreement_mask.bool() & ~direct
            radius = candidates & self.accepts(features, labels)
        eligible = direct | radius
        neighbor, checked = self.neighbor_accepts(features, labels, eligible)
        final = eligible & neighbor & checked
        return final, radius & final, final, checked


def scaled_class_thresholds(class_sum, class_count, threshold_lambda):
    """tau[c] = lambda * mean original-view max confidence for predicted class c.

    Missing classes use infinity and cannot pass the high-confidence path.
    No clipping or floor: lambda > 1 can intentionally produce tau > 1.
    """
    means = class_sum / class_count.clamp_min(1)
    thresholds = (means * float(threshold_lambda)).float()
    thresholds[class_count == 0] = float('inf')
    return thresholds, means.float()
