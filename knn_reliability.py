"""Prototype confidence regions and target-neighborhood support for OR rescue.

All statistics and scores are detached. Sigma is a pooled radial RMS scale,
not a per-coordinate Gaussian standard deviation or a calibrated probability.
Target memory contains every valid training feature, including low-confidence
predictions; confidence is checked after kNN retrieval, not before retrieval.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class KNNReliabilityBank(nn.Module):
    def __init__(self, num_classes, feature_dim, k=20, interval_lambda=1.5,
                 sigma_momentum=0.9, bandwidth_multiplier=1.0,
                 score_threshold=0.5, density_threshold=0.5,
                 query_chunk_size=128, eps=1e-8):
        super().__init__()
        if num_classes < 1 or feature_dim < 1 or k < 1 or query_chunk_size < 1:
            raise ValueError('class count, feature dimension, k and chunk size must be positive')
        for name, value in [('interval_lambda', interval_lambda),
                            ('bandwidth_multiplier', bandwidth_multiplier), ('eps', eps)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError('%s must be finite and positive' % name)
        if not 0 <= sigma_momentum < 1:
            raise ValueError('sigma_momentum must be in [0, 1)')
        if not 0 < score_threshold <= 1 or not 0 < density_threshold <= 1:
            raise ValueError('score and density thresholds must be in (0, 1]')
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.k = int(k)
        self.interval_lambda = float(interval_lambda)
        self.sigma_momentum = float(sigma_momentum)
        self.bandwidth_multiplier = float(bandwidth_multiplier)
        self.score_threshold = float(score_threshold)
        self.density_threshold = float(density_threshold)
        self.query_chunk_size = int(query_chunk_size)
        self.eps = float(eps)

        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer('prototype_initialized', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('global_var', torch.zeros(()))
        self.register_buffer('sigma_initialized', torch.tensor(False))
        self.register_buffer('source_count', torch.zeros((), dtype=torch.long))
        # Source accumulators and target memory are rebuilt, never restored as stale features.
        self.register_buffer('_source_sum', torch.zeros((), dtype=torch.float64), persistent=False)
        self.register_buffer('_source_count', torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer('memory_features', torch.empty(0, feature_dim), persistent=False)
        self.register_buffer('memory_labels', torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer('memory_confidences', torch.empty(0), persistent=False)
        self.register_buffer('memory_ids', torch.empty(0, dtype=torch.long), persistent=False)

    @property
    def sigma(self):
        return self.global_var.clamp_min(self.eps).sqrt()

    @torch.no_grad()
    def begin_source_refresh(self, prototypes, initialized):
        """Freeze scoring centers until the next refresh; retain the variance EMA."""
        if prototypes.shape != self.prototypes.shape or initialized.shape != self.prototype_initialized.shape:
            raise ValueError('prototype shapes do not match the reliability bank')
        prototypes = prototypes.detach().to(self.prototypes)
        valid = torch.isfinite(prototypes).all(dim=1) & (prototypes.norm(dim=1) > self.eps)
        self.prototypes.copy_(F.normalize(torch.nan_to_num(prototypes), dim=1))
        self.prototype_initialized.copy_(initialized.to(self.prototype_initialized) & valid)
        self._source_sum.zero_()
        self._source_count.zero_()
        # The refreshed centers must not be scored until the new scan is finalized.
        self.clear_target_memory()

    @torch.no_grad()
    def accumulate_source(self, features, labels):
        features = features.detach().to(self.prototypes)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > self.eps)
        valid &= (labels >= 0) & (labels < self.num_classes)
        features, labels = features[valid], labels[valid]
        ready = self.prototype_initialized[labels]
        features, labels = features[ready], labels[ready]
        if features.size(0) == 0:
            return
        z = F.normalize(features, dim=1)
        residual_sq = (z - self.prototypes[labels]).square().sum(dim=1)
        self._source_sum.add_(residual_sq.double().sum())
        self._source_count.add_(residual_sq.numel())

    @torch.no_grad()
    def finalize_source(self):
        """One sample-weighted global variance update per complete source scan."""
        self.source_count.copy_(self._source_count)
        if self._source_count.item() < 2:
            self.sigma_initialized.fill_(False)
            return
        observed_var = (self._source_sum / self._source_count).to(self.global_var)
        if not torch.isfinite(observed_var):
            self.sigma_initialized.fill_(False)
            return
        observed_var = observed_var.clamp_min(self.eps)
        if self.sigma_initialized.item():
            self.global_var.mul_(self.sigma_momentum).add_(
                observed_var, alpha=1.0 - self.sigma_momentum
            )
        else:
            self.global_var.copy_(observed_var)
        self.sigma_initialized.fill_(True)

    @torch.no_grad()
    def clear_target_memory(self):
        self.memory_features = self.prototypes.new_empty((0, self.feature_dim))
        self.memory_labels = self.prototype_initialized.new_empty((0,), dtype=torch.long)
        self.memory_confidences = self.global_var.new_empty((0,))
        self.memory_ids = self.memory_labels.clone()

    @torch.no_grad()
    def set_target_memory(self, features, labels, confidences, sample_ids):
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError('target memory must have shape [N, feature_dim]')
        n = features.size(0)
        if any(t.shape != (n,) for t in (labels, confidences, sample_ids)):
            raise ValueError('target memory vectors must have shape [N]')
        features = features.detach().to(self.prototypes)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        confidences = confidences.detach().to(self.global_var)
        sample_ids = sample_ids.detach().to(device=features.device, dtype=torch.long)
        if torch.unique(sample_ids).numel() != n:
            raise ValueError('target memory sample IDs must be unique')
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > self.eps)
        valid &= torch.isfinite(confidences) & (confidences >= 0) & (confidences <= 1)
        valid &= (labels >= 0) & (labels < self.num_classes)
        self.memory_features = F.normalize(features[valid], dim=1).contiguous()
        self.memory_labels = labels[valid].clone()
        self.memory_confidences = confidences[valid].clone()
        self.memory_ids = sample_ids[valid].clone()

    @torch.no_grad()
    def gate(self, features, pseudo_targets, sample_ids, candidate_mask, thresholds):
        """Score agreed candidates. Missing statistics/neighbors cannot rescue.

        density = mean(exp(-distance_sq / (2 * h^2)))
        score = self_in_region * mean(kernel * same_class * confident * in_region)
        h = bandwidth_multiplier * sigma; density is a support index, not a PDF.
        """
        features = features.detach().to(self.prototypes)
        n = features.size(0)
        if features.shape != (n, self.feature_dim):
            raise ValueError('query features must have shape [B, feature_dim]')
        device = features.device
        pseudo_targets = pseudo_targets.detach().to(device=device, dtype=torch.long)
        sample_ids = sample_ids.detach().to(device=device, dtype=torch.long)
        candidate_mask = candidate_mask.detach().to(device=device, dtype=torch.bool)
        if any(t.shape != (n,) for t in (pseudo_targets, sample_ids, candidate_mask)):
            raise ValueError('query vectors must have shape [B]')
        thresholds = thresholds.detach().to(self.global_var)
        if thresholds.shape != (self.num_classes,) or not torch.isfinite(thresholds).all():
            raise ValueError('thresholds must be a finite vector with one value per class')
        result = {
            'score': features.new_zeros(n),
            'density': features.new_zeros(n),
            'support_fraction': features.new_zeros(n),
            'checked_mask': torch.zeros(n, dtype=torch.bool, device=device),
            'dense_mask': torch.zeros(n, dtype=torch.bool, device=device),
            'pass_mask': torch.zeros(n, dtype=torch.bool, device=device),
        }
        if not self.sigma_initialized.item() or self.memory_features.size(0) < self.k:
            return result
        if not torch.isfinite(self.global_var) or self.global_var.item() <= 0:
            return result
        valid = candidate_mask & torch.isfinite(features).all(dim=1)
        valid &= features.norm(dim=1) > self.eps
        valid &= (pseudo_targets >= 0) & (pseudo_targets < self.num_classes)
        rows = valid.nonzero(as_tuple=False).flatten()
        rows = rows[self.prototype_initialized[pseudo_targets[rows]]]
        radius_sq = self.interval_lambda ** 2 * self.global_var
        h_sq = (self.bandwidth_multiplier ** 2 * self.global_var).clamp_min(self.eps)
        for start in range(0, rows.numel(), self.query_chunk_size):
            idx = rows[start:start + self.query_chunk_size]
            z = F.normalize(features[idx], dim=1)
            labels = pseudo_targets[idx]
            centers = self.prototypes[labels]
            # Allocate [chunk, N], never the full [N, N] target distance matrix.
            distances = (2.0 - 2.0 * z.mm(self.memory_features.t())).clamp_min(0)
            distances.masked_fill_(sample_ids[idx, None].eq(self.memory_ids[None, :]), float('inf'))
            nn_dist, nn_idx = distances.topk(self.k, dim=1, largest=False)
            enough_neighbors = torch.isfinite(nn_dist).all(dim=1)
            nn_labels = self.memory_labels[nn_idx]
            nn_features = self.memory_features[nn_idx]
            neighbor_in_region = (
                (nn_features - centers[:, None, :]).square().sum(dim=2) <= radius_sq
            )
            neighbor_confident = self.memory_confidences[nn_idx] >= thresholds[nn_labels]
            support = nn_labels.eq(labels[:, None]) & neighbor_confident & neighbor_in_region
            self_in_region = (z - centers).square().sum(dim=1) <= radius_sq
            weights = torch.exp(-nn_dist / (2.0 * h_sq))
            # Dividing by sum(weights) would remove the sparse-neighborhood penalty.
            score = (weights * support.float()).mean(dim=1) * self_in_region.float()
            density = weights.mean(dim=1)
            result['score'][idx] = score * enough_neighbors.float()
            result['density'][idx] = density * enough_neighbors.float()
            result['support_fraction'][idx] = support.float().mean(dim=1) * enough_neighbors.float()
            result['checked_mask'][idx] = enough_neighbors
        result['dense_mask'] = result['checked_mask'] & (result['density'] >= self.density_threshold)
        result['pass_mask'] = result['checked_mask'] & (result['score'] >= self.score_threshold)
        return result


def combine_rescue_masks(confidence_mask, agreement_mask, reliability_mask, enabled):
    """Keep v3's OR semantics and hard 0/1 masks used by CAST's affinity loss."""
    confidence_mask = confidence_mask.bool() & agreement_mask.bool()
    rescue_mask = torch.zeros_like(confidence_mask)
    if enabled:
        rescue_mask = agreement_mask.bool() & reliability_mask.bool() & ~confidence_mask
    return confidence_mask | rescue_mask, rescue_mask
