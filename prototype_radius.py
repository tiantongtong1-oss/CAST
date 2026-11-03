"""Fixed source prototypes with per-class mean-of-farthest-seven radii."""
import torch
from torch import nn
import torch.nn.functional as F


class PrototypeRadiusBank(nn.Module):
    def __init__(self, prototypes, initialized):
        super().__init__()
        self.register_buffer('prototypes', F.normalize(prototypes.detach().clone(), dim=1))
        self.register_buffer('initialized', initialized.detach().clone().bool())
        self.register_buffer('radii', prototypes.new_zeros(prototypes.size(0)))
        self.register_buffer('counts', torch.zeros(prototypes.size(0), dtype=torch.long,
                                                 device=prototypes.device))
        self.register_buffer('ready', torch.zeros_like(self.initialized))
        self.register_buffer('memory_features', prototypes.new_empty((0, prototypes.size(1))), persistent=False)
        self.register_buffer('memory_labels', torch.empty(0, dtype=torch.long, device=prototypes.device), persistent=False)
        self.register_buffer('_farthest', prototypes.new_full((prototypes.size(0), 7),
                                                            -float('inf')), persistent=False)

    @torch.no_grad()
    def accumulate_source(self, features, labels):
        features = features.detach().to(self.prototypes)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        z = F.normalize(torch.nan_to_num(features), dim=1)
        for c in range(self.prototypes.size(0)):
            if not self.initialized[c]:
                continue
            rows = valid & labels.eq(c)
            distances = (z[rows] - self.prototypes[c]).norm(dim=1)
            self.counts[c] += distances.numel()
            self._farthest[c] = torch.cat((self._farthest[c], distances)).topk(7).values

    @torch.no_grad()
    def finalize_source(self):
        self.ready.copy_(self.initialized & (self.counts >= 7))
        self.radii.zero_()
        self.radii[self.ready] = self._farthest[self.ready].mean(dim=1)

    @torch.no_grad()
    def accepts(self, features, labels):
        features = features.detach().to(self.prototypes)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid = (labels >= 0) & (labels < self.prototypes.size(0))
        valid &= torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        safe = labels.clamp(0, self.prototypes.size(0) - 1)
        z = F.normalize(torch.nan_to_num(features), dim=1)
        distances = (z - self.prototypes[safe]).norm(dim=1)
        return valid & self.ready[safe] & (distances < 0.7 * self.radii[safe])


    @torch.no_grad()
    def set_source_memory(self, features, labels):
        """Rebuild true-label source memory, prototypes and top-seven radii."""
        features = features.detach().to(self.prototypes)
        labels = labels.detach().to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        valid &= (labels >= 0) & (labels < self.prototypes.size(0))
        z = F.normalize(features[valid], dim=1)
        labels = labels[valid]
        self.prototypes.zero_()
        self.initialized.zero_()
        self.counts.zero_()
        self._farthest.fill_(-float('inf'))
        for c in range(self.prototypes.size(0)):
            rows = labels.eq(c)
            if rows.any():
                total = z[rows].sum(dim=0)
                if total.norm() > 1e-8:
                    self.prototypes[c] = F.normalize(total, dim=0)
                    self.initialized[c] = True
        self.accumulate_source(z, labels)
        self.finalize_source()
        # Non-persistent: always rebuild with the current teacher, never resume stale features.
        self.memory_features = z.contiguous()
        self.memory_labels = labels.clone()

    @torch.no_grad()
    def neighbor_accepts(self, features, labels, candidates, k=11, min_neighbors=3):
        """Unweighted strict majority of nearest true-label source neighbors."""
        features = features.detach().to(self.prototypes)
        labels = labels.to(device=features.device, dtype=torch.long)
        passed = torch.zeros_like(labels, dtype=torch.bool)
        checked = torch.zeros_like(passed)
        n = self.memory_features.size(0)
        take = min(k, n)
        if take < min_neighbors:
            return passed, checked
        valid = candidates.bool() & torch.isfinite(features).all(dim=1)
        valid &= features.norm(dim=1) > 1e-8
        valid &= (labels >= 0) & (labels < self.prototypes.size(0))
        rows = valid.nonzero(as_tuple=False).flatten()
        for chunk in rows.split(128):
            z = F.normalize(features[chunk], dim=1)
            indices = z.mm(self.memory_features.t()).topk(take, dim=1).indices
            votes = self.memory_labels[indices].eq(labels[chunk, None]).sum(dim=1)
            passed[chunk] = votes > take / 2
            checked[chunk] = True
        return passed, checked

    @torch.no_grad()
    def select(self, features, labels, confidence_mask, agreement_mask, enabled):
        direct = confidence_mask.bool() & agreement_mask.bool()
        radius = torch.zeros_like(direct)
        neighbor = torch.zeros_like(direct)
        checked = torch.zeros_like(direct)
        if enabled:
            candidates = agreement_mask.bool() & ~direct
            radius = candidates & self.accepts(features, labels)
            neighbor, checked = self.neighbor_accepts(features, labels, candidates & ~radius)
        return direct | radius | neighbor, radius, neighbor, checked
