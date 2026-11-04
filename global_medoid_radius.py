"""One source-sample medoid and one top-seven Euclidean radius for all classes."""
import torch
import torch.nn.functional as F
from prototype_radius import PrototypeRadiusBank


@torch.no_grad()
def exact_medoid_index(z, chunk_size=256):
    """Minimize sum of Euclidean (not squared) distances over all source samples.

    Exhaustively evaluates every candidate. Memory is O(chunk_size * N), not N².
    Float64 distance arithmetic reduces cancellation for near-identical vectors.
    Ties select the first source-memory row.
    """
    if z.ndim != 2 or z.size(0) == 0 or chunk_size < 1:
        raise ValueError('nonempty [N,D] features and positive chunk size required')
    z = z.double()
    scores = z.new_empty(z.size(0))
    for start in range(0, z.size(0), chunk_size):
        stop = min(start + chunk_size, z.size(0))
        distances = torch.cdist(z[start:stop], z, p=2,
                                compute_mode='use_mm_for_euclid_dist')
        local = torch.arange(stop - start, device=z.device)
        distances[local, local + start] = 0
        scores[start:stop] = distances.sum(dim=1)
    return scores.argmin()


class GlobalMedoidRadiusBank(PrototypeRadiusBank):
    def __init__(self, prototypes, initialized):
        super().__init__(prototypes, initialized)
        self.register_buffer('medoid', prototypes.new_zeros(prototypes.size(1)))
        self.register_buffer('global_radius', prototypes.new_zeros(()))
        self.register_buffer('global_ready', torch.tensor(False, device=prototypes.device))
        self.register_buffer('medoid_memory_index', torch.tensor(-1, device=prototypes.device))

    @torch.no_grad()
    def set_source_memory(self, features, labels):
        # Parent rebuilds true-label memory and class prototypes for the unchanged
        # prototype consistency loss. Class radii are not used by this selector.
        super().set_source_memory(features, labels)
        self.global_ready.fill_(False)
        self.global_radius.zero_()
        self.medoid.zero_()
        self.medoid_memory_index.fill_(-1)
        if self.memory_features.size(0) == 0:
            return
        index = exact_medoid_index(self.memory_features)
        self.medoid_memory_index.copy_(index)
        self.medoid.copy_(self.memory_features[index])
        if self.memory_features.size(0) < 7:
            return
        distances = (self.memory_features.double() - self.medoid.double()).norm(dim=1)
        self.global_radius.copy_(distances.topk(7).values.mean())
        self.global_ready.fill_(True)

    @torch.no_grad()
    def accepts(self, features, labels):
        features = features.detach().to(self.medoid)
        labels = labels.to(device=features.device, dtype=torch.long)
        valid = torch.isfinite(features).all(dim=1) & (features.norm(dim=1) > 1e-8)
        valid &= (labels >= 0) & (labels < self.prototypes.size(0))
        z = F.normalize(torch.nan_to_num(features), dim=1)
        distances = (z.double() - self.medoid.double()).norm(dim=1)
        return valid & self.global_ready & (distances < 0.7 * self.global_radius.double())
