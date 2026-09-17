import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassDistributionBank(nn.Module):
    """Source class distributions used to validate target pseudo labels.

    For each class c, the bank estimates a mean p_c and a shrinkage covariance
    Sigma_c from normalized source features. A target feature z receives:

        D_c(z) = (z - p_c)^T Sigma_c^{-1} (z - p_c)

    and a source-supported local density estimated with a Gaussian kernel in
    the same Mahalanobis geometry. The gate uses log energy

        log E_c(z) = log(D_c(z) + eps) - log L_c(z)

    which is monotonic with E_c(z) = D_c(z) / (L_c(z) + eps) while avoiding
    high-dimensional numerical underflow. Per-class thresholds are calibrated
    from leave-one-out source energies using a configurable quantile.

    The density memory is capped to keep the per-batch gate practical. Mean and
    covariance still use every deterministic source feature observed during a
    rebuild. ``bandwidth`` is dimension-normalized: the effective kernel h is
    bandwidth * sqrt(feature_dim), which keeps Mahalanobis kernels usable in a
    512-D representation space.
    """

    def __init__(self, num_classes, feature_dim, bandwidth=1.0,
                 covariance_shrinkage=0.05, energy_quantile=0.95,
                 max_density_samples=256, eps=1e-6):
        super().__init__()
        if bandwidth <= 0.0:
            raise ValueError('energy bandwidth must be positive')
        if not (0.0 <= covariance_shrinkage <= 1.0):
            raise ValueError('covariance_shrinkage must be in [0, 1]')
        if not (0.0 < energy_quantile < 1.0):
            raise ValueError('energy_quantile must be in (0, 1)')
        if max_density_samples < 2:
            raise ValueError('max_density_samples must be at least 2')
        if eps <= 0.0:
            raise ValueError('eps must be positive')

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.bandwidth = float(bandwidth)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.energy_quantile = float(energy_quantile)
        self.max_density_samples = int(max_density_samples)
        self.eps = float(eps)

        self.register_buffer(
            'source_means',
            torch.zeros(self.num_classes, self.feature_dim),
        )
        self.register_buffer(
            'source_precisions',
            torch.zeros(self.num_classes, self.feature_dim, self.feature_dim),
        )
        self.register_buffer(
            'source_counts',
            torch.zeros(self.num_classes, dtype=torch.long),
        )
        self.register_buffer(
            'log_energy_thresholds',
            torch.full((self.num_classes,), float('inf')),
        )
        self.register_buffer(
            'source_initialized',
            torch.zeros(self.num_classes, dtype=torch.bool),
        )

        self._source_chunks = [[] for _ in range(self.num_classes)]
        self._density_features = [None for _ in range(self.num_classes)]
        self._density_projected = [None for _ in range(self.num_classes)]
        self._density_quadratic = [None for _ in range(self.num_classes)]

    @torch.no_grad()
    def reset_source(self):
        self.source_means.zero_()
        self.source_precisions.zero_()
        self.source_counts.zero_()
        self.log_energy_thresholds.fill_(float('inf'))
        self.source_initialized.zero_()
        self._source_chunks = [[] for _ in range(self.num_classes)]
        self._density_features = [None for _ in range(self.num_classes)]
        self._density_projected = [None for _ in range(self.num_classes)]
        self._density_quadratic = [None for _ in range(self.num_classes)]

    @torch.no_grad()
    def accumulate_source(self, features, labels):
        features = F.normalize(features.detach(), dim=1)
        labels = labels.detach().long()
        for c in range(self.num_classes):
            class_mask = labels.eq(c)
            if class_mask.any():
                self._source_chunks[c].append(features[class_mask].clone())

    @torch.no_grad()
    def finalize_source(self):
        identity = torch.eye(
            self.feature_dim,
            device=self.source_means.device,
            dtype=self.source_means.dtype,
        )

        for c in range(self.num_classes):
            if not self._source_chunks[c]:
                continue

            features = torch.cat(self._source_chunks[c], dim=0)
            count = int(features.size(0))
            self.source_counts[c] = count
            if count < 2:
                continue

            mean = features.mean(dim=0)
            centered = features - mean.unsqueeze(0)
            covariance = centered.t().mm(centered) / float(max(count - 1, 1))

            scale = torch.trace(covariance) / float(self.feature_dim)
            scale = scale.clamp_min(self.eps)
            covariance = (
                (1.0 - self.covariance_shrinkage) * covariance
                + self.covariance_shrinkage * scale * identity
                + self.eps * identity
            )

            # The covariance is symmetric positive definite after shrinkage and
            # jitter. Eigh is more stable here than a direct matrix inverse.
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            eigenvalues = eigenvalues.clamp_min(self.eps)
            precision = (
                eigenvectors * eigenvalues.reciprocal().unsqueeze(0)
            ).mm(eigenvectors.t())

            self.source_means[c].copy_(mean)
            self.source_precisions[c].copy_(precision)

            memory_size = min(count, self.max_density_samples)
            if memory_size == count:
                representatives = features
            else:
                indices = torch.linspace(
                    0,
                    count - 1,
                    steps=memory_size,
                    device=features.device,
                ).round().long()
                representatives = features.index_select(0, indices)

            representatives = representatives.contiguous()
            projected = representatives.mm(precision)
            quadratic = (projected * representatives).sum(dim=1)
            self._density_features[c] = representatives
            self._density_projected[c] = projected
            self._density_quadratic[c] = quadratic

            threshold = self._calibrate_class_threshold(c)
            self.log_energy_thresholds[c] = threshold
            self.source_initialized[c] = torch.isfinite(threshold)

        # Release full source feature chunks after statistics are finalized.
        self._source_chunks = [[] for _ in range(self.num_classes)]

    def _effective_bandwidth(self):
        return self.bandwidth * math.sqrt(float(self.feature_dim))

    @torch.no_grad()
    def _mahalanobis_to_mean(self, features, class_index):
        centered = features - self.source_means[class_index].unsqueeze(0)
        projected = centered.mm(self.source_precisions[class_index])
        return (projected * centered).sum(dim=1).clamp_min(0.0)

    @torch.no_grad()
    def _pairwise_mahalanobis(self, features, class_index):
        representatives = self._density_features[class_index]
        if representatives is None:
            return features.new_empty((features.size(0), 0))

        precision = self.source_precisions[class_index]
        feature_projected = features.mm(precision)
        feature_quadratic = (feature_projected * features).sum(dim=1)
        representative_quadratic = self._density_quadratic[class_index]
        cross = feature_projected.mm(representatives.t())
        distances = (
            feature_quadratic.unsqueeze(1)
            + representative_quadratic.unsqueeze(0)
            - 2.0 * cross
        )
        return distances.clamp_min(0.0)

    @torch.no_grad()
    def _log_density(self, pairwise_distances, exclude_diagonal=False):
        memory_size = int(pairwise_distances.size(1))
        if memory_size == 0:
            return pairwise_distances.new_full(
                (pairwise_distances.size(0),), float('-inf')
            )

        h = self._effective_bandwidth()
        log_kernel = -pairwise_distances / (2.0 * h * h)

        if exclude_diagonal:
            if pairwise_distances.size(0) != memory_size or memory_size < 2:
                return pairwise_distances.new_full(
                    (pairwise_distances.size(0),), float('-inf')
                )
            diagonal = torch.eye(
                memory_size,
                device=pairwise_distances.device,
                dtype=torch.bool,
            )
            log_kernel = log_kernel.masked_fill(diagonal, float('-inf'))
            normalizer = float(memory_size - 1)
        else:
            normalizer = float(memory_size)

        return torch.logsumexp(log_kernel, dim=1) - math.log(normalizer)

    @torch.no_grad()
    def _calibrate_class_threshold(self, class_index):
        representatives = self._density_features[class_index]
        if representatives is None or representatives.size(0) < 2:
            return self.source_means.new_tensor(float('inf'))

        global_deviation = self._mahalanobis_to_mean(
            representatives, class_index
        )
        pairwise = self._pairwise_mahalanobis(representatives, class_index)
        log_density = self._log_density(pairwise, exclude_diagonal=True)
        log_energy = torch.log(global_deviation + self.eps) - log_density
        finite = log_energy[torch.isfinite(log_energy)]
        if finite.numel() == 0:
            return self.source_means.new_tensor(float('inf'))
        return torch.quantile(finite, self.energy_quantile)

    @torch.no_grad()
    def gate(self, features, labels, candidate_mask):
        """Return per-sample log energy and a source-distribution pass mask.

        ``candidate_mask`` should normally be the existing dual-view confidence
        mask. Classes without a valid source distribution fail open rather than
        deleting all pseudo labels from that class.
        """
        features = F.normalize(features.detach(), dim=1)
        labels = labels.detach().long()
        candidate_mask = candidate_mask.bool()

        log_energy = features.new_full((features.size(0),), float('nan'))
        pass_mask = candidate_mask.clone()

        for c in range(self.num_classes):
            class_mask = candidate_mask & labels.eq(c)
            if not class_mask.any():
                continue
            if not bool(self.source_initialized[c].item()):
                continue

            class_features = features[class_mask]
            global_deviation = self._mahalanobis_to_mean(class_features, c)
            pairwise = self._pairwise_mahalanobis(class_features, c)
            log_density = self._log_density(pairwise, exclude_diagonal=False)
            class_log_energy = (
                torch.log(global_deviation + self.eps) - log_density
            )

            log_energy[class_mask] = class_log_energy
            pass_mask[class_mask] = (
                class_log_energy <= self.log_energy_thresholds[c]
            )

        return log_energy, pass_mask
