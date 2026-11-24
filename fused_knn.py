"""Equal-weight source true-label and target teacher-probability kNN fusion."""
import torch
from emotion_radius import grouped_radii
from torch import nn
import torch.nn.functional as F


class FusedKNNBank(nn.Module):
    def __init__(self, feature_dim, num_classes=7, support_threshold=0.7, k=11):
        super().__init__()
        if not 0 <= support_threshold <= 1 or k < 1:
            raise ValueError('support threshold must be in [0,1] and k positive')
        self.register_buffer('class_prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer('class_center_ready', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('class_counts', torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer('group_radii', torch.zeros(2, dtype=torch.float64))
        self.register_buffer('group_radius_ready', torch.zeros(2, dtype=torch.bool))
        # 0: surprise/happy/neutral; 1: fear/disgust/sad/angry.
        self.register_buffer('class_radius_group', torch.tensor(
            [0 if c in (0, 3, 6) else 1 for c in range(num_classes)], dtype=torch.long))
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
        self.group_radii.zero_()
        self.group_radius_ready.zero_()
        self.class_prototypes.zero_()
        self.class_center_ready.zero_()
        self.class_counts.zero_()
        class_distances_for_groups = {}
        for c in range(self.num_classes):
            z = self.source_features[self.source_labels.eq(c)]
            self.class_counts[c] = z.size(0)
            if z.size(0) < 2:
                continue
            # Same definition as the source prototypes in PrototypeBank.
            feature_sum = z.sum(dim=0)
            if feature_sum.norm().item() <= 1e-8:
                continue
            prototype = F.normalize(feature_sum, dim=0)
            self.class_prototypes[c].copy_(prototype)
            self.class_center_ready[c] = True
            class_distances_for_groups[c] = (z.double() - prototype.double()).norm(dim=1)

        radii = grouped_radii({
            c: distances.cpu().tolist() for c, distances in class_distances_for_groups.items()
        })
        for group, radius in enumerate(radii):
            if radius is not None:
                self.group_radii[group] = radius
                self.group_radius_ready[group] = True

        for c in range(self.num_classes):
            count = int(self.class_counts[c].item())
            if c not in class_distances_for_groups:
                print(f'[Source Class {c}] N={count} CenterReady=False Coverage_R=N/A')
                continue

            class_distances = class_distances_for_groups[c]
            # These per-class quantiles are for diagnostics only.
            quantiles = torch.quantile(
                class_distances,
                class_distances.new_tensor([0.50, 0.85, 0.90]),
                interpolation='linear',
            )
            group_radius = self.group_radii[self.class_radius_group[c]]
            inside = int((class_distances < group_radius).sum().item())
            print(
                f'【源域类别 Source Class {c}】'
                f'有效样本数 N={count}，'
                f'距离中位数 D50={quantiles[0].item():.6f}，'
                f'距离85%分位数 D85={quantiles[1].item():.6f}，'
                f'距离90%分位数 D90={quantiles[2].item():.6f}，'
                f'分组半径 R={group_radius.item():.6f}，'
                f'半径内样本数 Inside_R={inside}，'
                f'半径覆盖率 Coverage_R={inside / count:.2%}'
            )

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
        features = features.detach().to(self.class_prototypes)
        labels = labels.to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        valid &= (labels >= 0) & (labels < self.num_classes)
        safe = labels.clamp(0, self.num_classes - 1)
        z = F.normalize(torch.nan_to_num(features), dim=1)
        distances = (z.double() - self.class_prototypes[safe].double()).norm(dim=1)
        groups = self.class_radius_group[safe]
        return (valid & self.group_radius_ready[groups] & self.class_center_ready[safe]
                & (distances < self.group_radii[groups]))

    @torch.no_grad()
    def select(self, features, labels, confidence_mask, agreement_mask, sample_ids,
               radius_candidates=None):
        """Shared support threshold; radius-only candidates require both domains to agree.

        Omitted radius_candidates preserves the legacy confidence-only call API.
        """
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
        confidence_mask = confidence_mask.to(device=features.device, dtype=torch.bool)
        radius_candidates = (torch.zeros_like(confidence_mask) if radius_candidates is None
                             else radius_candidates.to(device=features.device, dtype=torch.bool))
        # Confidence path wins on overlap; rescue is restricted to low confidence.
        radius_candidates = radius_candidates & ~confidence_mask
        eligible = (confidence_mask | radius_candidates) & agreement_mask.bool()
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
            fusion_pass = ready & mixed.argmax(dim=1).eq(labels[rows]) & (s >= self.support_threshold)
            both_domains_agree = (ps.argmax(dim=1).eq(labels[rows])
                                  & pt.argmax(dim=1).eq(labels[rows]))
            high_pass = confidence_mask[rows] & fusion_pass
            rescue_pass = radius_candidates[rows] & fusion_pass & both_domains_agree
            passed[rows] = high_pass | rescue_pass
        return passed, checked, support, source_support, target_support, disagreement
