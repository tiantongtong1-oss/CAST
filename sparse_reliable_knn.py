"""v8: stable senders and support-based rescue independent of absolute density.

The v7 scorer remains available for ablation. Gaussian membership is still a
source-only heuristic, not a calibrated probability guarantee under domain shift.
"""

import math
import torch
import torch.nn.functional as F

from global_gaussian_knn_reliability import GlobalGaussianKNNReliabilityBank


class SparseReliableKNNBank(GlobalGaussianKNNReliabilityBank):
    def __init__(self, *args, score_mode='support', min_support=3,
                 min_effective_support=2.0, min_purity=0.8,
                 min_margin=0.2, radius_multiplier=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        if score_mode not in ('v7', 'support'):
            raise ValueError('unknown score mode')
        if min_support < 1 or (score_mode == 'support' and min_support > self.k):
            raise ValueError('min_support must be positive and <= k for support mode')
        if not math.isfinite(min_effective_support) or not 1 <= min_effective_support <= self.k:
            raise ValueError('min_effective_support must be finite and in [1, k]')
        if not 0 < min_purity <= 1 or not 0 <= min_margin <= 1:
            raise ValueError('invalid purity or margin')
        if not math.isfinite(radius_multiplier) or radius_multiplier <= 0:
            raise ValueError('radius_multiplier must be finite and positive')
        self.score_mode = score_mode
        self.min_support = int(min_support)
        self.min_effective_support = float(min_effective_support)
        self.min_purity = float(min_purity)
        self.min_margin = float(min_margin)
        self.radius_multiplier = float(radius_multiplier)
        self.register_buffer('memory_sender_scores', torch.empty(0), persistent=False)

    @torch.no_grad()
    def set_sender_scores(self, scores):
        scores = scores.detach().to(self.global_var)
        if scores.shape != self.memory_confidences.shape:
            raise ValueError('sender scores must align with filtered memory IDs')
        if not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('sender scores must be finite and in [0, 1]')
        self.memory_sender_scores = scores.clone()

    @torch.no_grad()
    def set_target_memory(self, features, labels, confidences, sample_ids):
        super().set_target_memory(features, labels, confidences, sample_ids)
        # Explicit initialization prevents stale sender roles across refreshes.
        self.memory_sender_scores = torch.zeros_like(self.memory_confidences)

    @torch.no_grad()
    def gate(self, features, pseudo_targets, sample_ids, candidate_mask, thresholds):
        if self.score_mode == 'v7':
            return super().gate(features, pseudo_targets, sample_ids, candidate_mask, thresholds)
        features = features.detach().to(self.class_means)
        n = features.size(0)
        if features.shape != (n, self.feature_dim):
            raise ValueError('invalid query shape')
        labels = pseudo_targets.detach().to(device=features.device, dtype=torch.long)
        ids = sample_ids.detach().to(device=features.device, dtype=torch.long)
        candidates = candidate_mask.detach().to(device=features.device, dtype=torch.bool)
        if any(v.shape != (n,) for v in (labels, ids, candidates)):
            raise ValueError('query vectors must have matching shapes')
        zero = features.new_zeros(n)
        false = torch.zeros(n, dtype=torch.bool, device=features.device)
        result = {name: zero.clone() for name in (
            'score', 'density', 'weighted_support', 'support_fraction',
            'support_count', 'effective_support', 'neighbor_margin')}
        result.update({name: false.clone() for name in (
            'checked_mask', 'dense_mask', 'sparse_mask', 'pass_mask',
            'in_distribution', 'neighbor_reliable')})
        result['mahalanobis_sq'] = features.new_full((n,), float('inf'))
        result['neighbor_probs'] = features.new_zeros(n, self.num_classes)
        if not self.sigma_initialized.item() or self.memory_ids.numel() < 2:
            return result
        if self.memory_sender_scores.shape != self.memory_confidences.shape:
            raise RuntimeError('refresh sender scores before querying memory')

        valid = candidates & torch.isfinite(features).all(1) & (features.norm(dim=1) > self.eps)
        valid &= (labels >= 0) & (labels < self.num_classes)
        rows = valid.nonzero(as_tuple=False).flatten()
        rows = rows[self.distribution_initialized[labels[rows]]]
        k = min(self.k, self.memory_ids.numel())
        memory_region = self.in_distribution(self.memory_features, self.memory_labels)
        for start in range(0, rows.numel(), self.query_chunk_size):
            idx = rows[start:start + self.query_chunk_size]
            z = F.normalize(features[idx], dim=1)
            # A sender contributes on its own class scale, including competing
            # classes in the vote used for purity and optional soft labels.
            vector_var = (self.feature_dim * self.class_var[self.memory_labels]).clamp_min(self.eps)
            h_sq = self.bandwidth_multiplier ** 2 * vector_var
            radius_sq = 2.0 * self.radius_multiplier * vector_var
            distance = (2 - 2 * z.mm(self.memory_features.t())).clamp_min(0)
            distance.masked_fill_(ids[idx, None].eq(self.memory_ids[None, :]), float('inf'))
            nn_dist, nn_idx = distance.topk(k, dim=1, largest=False)
            finite = torch.isfinite(nn_dist)
            weights = torch.exp(-nn_dist / (2 * h_sq[nn_idx]))
            nn_labels = self.memory_labels[nn_idx]
            reliability = self.memory_sender_scores[nn_idx]
            reliable = finite & (nn_dist <= radius_sq[nn_idx]) & memory_region[nn_idx]
            reliable &= reliability > 0
            vote_weights = weights * reliability * reliable.float()
            votes = features.new_zeros(idx.numel(), self.num_classes)
            votes.scatter_add_(1, nn_labels, vote_weights)
            vote_sum = votes.sum(1)
            probs = votes / vote_sum[:, None].clamp_min(self.eps)
            winning_prob, winning_class = probs.max(1)
            if self.num_classes > 1:
                margin = probs.topk(2, dim=1).values.diff(dim=1).neg().squeeze(1)
            else:
                margin = winning_prob

            query_class = labels[idx]
            matches = nn_labels.eq(query_class[:, None]) & reliable
            support_weights = vote_weights * matches.float()
            count = matches.sum(1)
            effective = support_weights.sum(1).square() / support_weights.square().sum(1).clamp_min(self.eps)
            purity = probs.gather(1, query_class[:, None]).squeeze(1)
            # Mean reliability of agreeing senders. A single high-confidence
            # neighbor cannot satisfy count/effective-support requirements.
            quality = support_weights.sum(1) / (weights * matches.float()).sum(1).clamp_min(self.eps)
            self_region = self.in_distribution(z, query_class)
            score = purity * quality * self_region.float()
            enough = finite.sum(1) >= self.min_support
            accepted = enough & self_region & winning_class.eq(query_class)
            accepted &= (count >= self.min_support) & (effective >= self.min_effective_support)
            accepted &= (purity >= self.min_purity) & (margin >= self.min_margin)
            accepted &= (vote_sum > self.eps) & (score >= self.score_threshold)

            # Separate evidence for an OPTIONAL conflicting soft-label recipient.
            winning_matches = nn_labels.eq(winning_class[:, None]) & reliable
            winning_weights = vote_weights * winning_matches.float()
            winning_effective = winning_weights.sum(1).square() / winning_weights.square().sum(1).clamp_min(self.eps)
            winning_quality = winning_weights.sum(1) / (weights * winning_matches.float()).sum(1).clamp_min(self.eps)
            neighbor_good = enough & (winning_matches.sum(1) >= self.min_support)
            neighbor_good &= (winning_effective >= self.min_effective_support)
            neighbor_good &= (winning_prob >= self.min_purity) & (margin >= self.min_margin)
            neighbor_good &= (vote_sum > self.eps) & (winning_prob * winning_quality >= self.score_threshold)
            neighbor_good &= self.in_distribution(z, winning_class)

            density = weights.sum(1) / finite.sum(1).clamp_min(1)
            result['score'][idx] = score
            result['density'][idx] = density
            result['weighted_support'][idx] = purity
            result['support_fraction'][idx] = count / finite.sum(1).clamp_min(1)
            result['support_count'][idx] = count.to(z.dtype)
            result['effective_support'][idx] = effective
            result['neighbor_margin'][idx] = margin
            result['neighbor_probs'][idx] = probs
            result['neighbor_reliable'][idx] = neighbor_good
            result['checked_mask'][idx] = enough
            result['in_distribution'][idx] = self_region
            result['mahalanobis_sq'][idx] = self.distribution_distance(z, query_class)
            result['dense_mask'][idx] = enough & (density >= self.density_threshold)
            result['sparse_mask'][idx] = enough & (density < self.density_threshold)
            result['pass_mask'][idx] = accepted
        return result

