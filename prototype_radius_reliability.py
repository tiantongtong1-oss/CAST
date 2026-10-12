"""Source-only prototypes and mean of the farthest-k Euclidean distances.

Historical Gaussian experiments live in global_gaussian_knn_reliability.py;
this bank does not calculate variances or Mahalanobis distances.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class PrototypeRadiusKNNReliabilityBank(nn.Module):
    def __init__(self, num_classes, feature_dim, k=20, source_radius_k=11,
                 bandwidth_multiplier=1.0, score_threshold=0.5,
                 density_threshold=0.5, query_chunk_size=128,
                 sparse_penalty=0.5, eps=1e-8):
        super().__init__()
        for name, value in (("num_classes", num_classes), ("feature_dim", feature_dim),
                            ("k", k), ("source_radius_k", source_radius_k),
                            ("query_chunk_size", query_chunk_size)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(name + " must be a positive integer")
        for name, value in (("bandwidth_multiplier", bandwidth_multiplier), ("eps", eps)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if not 0 < score_threshold <= 1 or not 0 < density_threshold <= 1:
            raise ValueError("score and density thresholds must be in (0, 1]")
        if not 0 < sparse_penalty <= 1:
            raise ValueError("sparse_penalty must be in (0, 1]")
        self.num_classes, self.feature_dim = int(num_classes), int(feature_dim)
        self.k, self.source_radius_k = int(k), int(source_radius_k)
        self.bandwidth_multiplier = float(bandwidth_multiplier)
        self.score_threshold, self.density_threshold = score_threshold, density_threshold
        self.query_chunk_size, self.sparse_penalty, self.eps = int(query_chunk_size), sparse_penalty, eps
        self.register_buffer("class_means", torch.zeros(num_classes, feature_dim))
        self.register_buffer("class_radii", torch.zeros(num_classes))
        self.register_buffer("distribution_initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("class_source_counts", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("source_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("refresh_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("_source_sum", torch.zeros(num_classes, feature_dim, dtype=torch.float64), persistent=False)
        self.register_buffer("_source_count", torch.zeros(num_classes, dtype=torch.long), persistent=False)
        self.register_buffer("memory_features", torch.empty(0, feature_dim), persistent=False)
        self.register_buffer("memory_labels", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("memory_confidences", torch.empty(0), persistent=False)
        self.register_buffer("memory_ids", torch.empty(0, dtype=torch.long), persistent=False)
        # CPU snapshots avoid retaining the whole source dataset on GPU. Distances
        # must be computed after the final class means are known (not batch means).
        self._source_batches = []

    @torch.no_grad()
    def begin_source_refresh(self):
        self._source_sum.zero_()
        self._source_count.zero_()
        self._source_batches.clear()
        self.clear_target_memory()

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

    @torch.no_grad()
    def clear_target_memory(self):
        self.memory_features = self.class_means.new_empty(
            (0, self.feature_dim)
        )
        self.memory_labels = self.distribution_initialized.new_empty(
            (0,), dtype=torch.long
        )
        self.memory_confidences = self.class_radii.new_empty((0,))
        self.memory_ids = self.memory_labels.clone()

    @torch.no_grad()
    def set_target_memory(
        self, features, labels, confidences, sample_ids
    ):
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError(
                "target memory must have shape [N, feature_dim]"
            )
        n = features.size(0)
        if any(
            t.shape != (n,)
            for t in (labels, confidences, sample_ids)
        ):
            raise ValueError("target memory vectors must have shape [N]")

        features = features.detach().to(self.class_means)
        labels = labels.detach().to(
            device=features.device, dtype=torch.long
        )
        confidences = confidences.detach().to(self.class_radii)
        sample_ids = sample_ids.detach().to(
            device=features.device, dtype=torch.long
        )
        if torch.unique(sample_ids).numel() != n:
            raise ValueError("target memory sample IDs must be unique")

        valid = torch.isfinite(features).all(dim=1)
        valid &= features.norm(dim=1) > self.eps
        valid &= torch.isfinite(confidences)
        valid &= (confidences >= 0) & (confidences <= 1)
        valid &= (labels >= 0) & (labels < self.num_classes)

        self.memory_features = F.normalize(
            features[valid], dim=1
        ).contiguous()
        self.memory_labels = labels[valid].clone()
        self.memory_confidences = confidences[valid].clone()
        self.memory_ids = sample_ids[valid].clone()

    @torch.no_grad()
    def gate(
        self,
        features,
        pseudo_targets,
        sample_ids,
        candidate_mask,
        thresholds,
    ):
        """Score pseudo-label reliability using prototype-radius membership + kNN support.

        weighted_support = sum_j(w_j * support_j) / sum_j(w_j)
        density = mean_j(w_j)
        density_factor = density if dense else density * sparse_penalty
        score = query_in_region * weighted_support * density_factor

        support_j requires same pseudo class, confidence, and membership in the
        candidate class source prototype region.
        """
        features = features.detach().to(self.class_means)
        n = features.size(0)
        if features.shape != (n, self.feature_dim):
            raise ValueError(
                "query features must have shape [B, feature_dim]"
            )

        device = features.device
        pseudo_targets = pseudo_targets.detach().to(
            device=device, dtype=torch.long
        )
        sample_ids = sample_ids.detach().to(
            device=device, dtype=torch.long
        )
        candidate_mask = candidate_mask.detach().to(
            device=device, dtype=torch.bool
        )
        if any(
            t.shape != (n,)
            for t in (pseudo_targets, sample_ids, candidate_mask)
        ):
            raise ValueError("query vectors must have shape [B]")

        thresholds = thresholds.detach().to(self.class_radii)
        if (
            thresholds.shape != (self.num_classes,)
            or not torch.isfinite(thresholds).all()
        ):
            raise ValueError(
                "thresholds must be a finite vector with one value per class"
            )

        result = {
            "score": features.new_zeros(n),
            "prototype_distance": features.new_full((n,), float("inf")),
            "in_distribution": torch.zeros(
                n, dtype=torch.bool, device=device
            ),
            "density": features.new_zeros(n),
            "weighted_support": features.new_zeros(n),
            "support_fraction": features.new_zeros(n),
            "checked_mask": torch.zeros(
                n, dtype=torch.bool, device=device
            ),
            "dense_mask": torch.zeros(
                n, dtype=torch.bool, device=device
            ),
            "sparse_mask": torch.zeros(
                n, dtype=torch.bool, device=device
            ),
            "pass_mask": torch.zeros(
                n, dtype=torch.bool, device=device
            ),
        }

        if (
            not self.distribution_initialized.any().item()
            or self.memory_features.size(0) < self.k
        ):
            return result
        valid = candidate_mask & torch.isfinite(features).all(dim=1)
        valid &= features.norm(dim=1) > self.eps
        valid &= (pseudo_targets >= 0) & (
            pseudo_targets < self.num_classes
        )
        rows = valid.nonzero(as_tuple=False).flatten()
        rows = rows[
            self.distribution_initialized[pseudo_targets[rows]]
        ]

        for start in range(0, rows.numel(), self.query_chunk_size):
            idx = rows[start : start + self.query_chunk_size]
            z = F.normalize(features[idx], dim=1)
            labels = pseudo_targets[idx]
            h_sq = (self.bandwidth_multiplier ** 2 * self.class_radii[labels].square()).clamp_min(self.eps)[:, None]

            distances = (
                2.0 - 2.0 * z.mm(self.memory_features.t())
            ).clamp_min(0)
            distances.masked_fill_(
                sample_ids[idx, None].eq(
                    self.memory_ids[None, :]
                ),
                float("inf"),
            )
            nn_dist, nn_idx = distances.topk(
                self.k, dim=1, largest=False
            )
            enough_neighbors = torch.isfinite(nn_dist).all(dim=1)

            nn_labels = self.memory_labels[nn_idx]
            nn_features = self.memory_features[nn_idx]

            # Every neighbor is tested against the candidate pseudo-label class
            # prototype region, not its own predicted-class prototype region.
            candidate_labels = labels[:, None].expand_as(nn_labels)
            neighbor_in_region = self.in_distribution(
                nn_features, candidate_labels
            )
            neighbor_confident = (
                self.memory_confidences[nn_idx]
                >= thresholds[nn_labels]
            )
            support = (
                nn_labels.eq(candidate_labels)
                & neighbor_confident
                & neighbor_in_region
            )

            prototype_distance = self.distribution_distance(
                z, labels
            )
            self_in_region = (
                self.in_distribution(z, labels)
            )

            weights = torch.exp(
                -nn_dist / (2.0 * h_sq)
            )
            finite_weights = torch.where(
                torch.isfinite(weights),
                weights,
                torch.zeros_like(weights),
            )
            density = finite_weights.mean(dim=1)
            weighted_support = (
                (finite_weights * support.float()).sum(dim=1)
                / finite_weights.sum(dim=1).clamp_min(self.eps)
            )
            dense = density >= self.density_threshold
            density_factor = torch.where(
                dense,
                density,
                density * self.sparse_penalty,
            )
            score = (
                self_in_region.float()
                * weighted_support
                * density_factor
                * enough_neighbors.float()
            )

            result["prototype_distance"][idx] = prototype_distance
            result["in_distribution"][idx] = self_in_region
            result["score"][idx] = score
            result["density"][idx] = (
                density * enough_neighbors.float()
            )
            result["weighted_support"][idx] = (
                weighted_support * enough_neighbors.float()
            )
            result["support_fraction"][idx] = (
                support.float().mean(dim=1)
                * enough_neighbors.float()
            )
            result["checked_mask"][idx] = enough_neighbors
            result["dense_mask"][idx] = (
                enough_neighbors & dense
            )
            result["sparse_mask"][idx] = (
                enough_neighbors & ~dense
            )

        result["pass_mask"] = (
            result["checked_mask"]
            & result["in_distribution"]
            & (result["score"] >= self.score_threshold)
        )
        return result


def combine_rescue_masks(
    confidence_mask,
    agreement_mask,
    reliability_mask,
    enabled,
):
    confidence_mask = confidence_mask.bool() & agreement_mask.bool()
    rescue_mask = torch.zeros_like(confidence_mask)
    if enabled:
        rescue_mask = (
            agreement_mask.bool()
            & reliability_mask.bool()
            & ~confidence_mask
        )
    return confidence_mask | rescue_mask, rescue_mask



