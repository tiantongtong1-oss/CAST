import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureHook:
    """Capture the exact feature tensor produced by a module during forward."""

    def __init__(self, module):
        self.output = None
        self._handle = module.register_forward_hook(self._save_output)

    def _save_output(self, module, inputs, output):
        del module, inputs
        self.output = output

    def close(self):
        self._handle.remove()
        self.output = None


class PrototypeBank(nn.Module):
    """Source-anchored EMA class prototypes for target consistency training.

    Source prototypes are computed once from labeled RAF-DB features and remain
    fixed semantic anchors. Target prototypes are updated class-by-class from
    reliable dual-view EMA-teacher features. The prototype used by the loss is
    a normalized blend of the fixed source anchor and the adaptive target EMA.
    """

    def __init__(self, num_classes, feature_dim, momentum=0.99,
                 source_anchor=0.5):
        super().__init__()
        if not (0.0 <= momentum < 1.0):
            raise ValueError('prototype momentum must be in [0, 1)')
        if not (0.0 <= source_anchor <= 1.0):
            raise ValueError('source_anchor must be in [0, 1]')

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.source_anchor = float(source_anchor)

        self.register_buffer(
            'source_feature_sums',
            torch.zeros(self.num_classes, self.feature_dim),
        )
        self.register_buffer(
            'source_counts',
            torch.zeros(self.num_classes, dtype=torch.long),
        )
        self.register_buffer(
            'source_prototypes',
            torch.zeros(self.num_classes, self.feature_dim),
        )
        self.register_buffer(
            'source_initialized',
            torch.zeros(self.num_classes, dtype=torch.bool),
        )
        self.register_buffer(
            'target_prototypes',
            torch.zeros(self.num_classes, self.feature_dim),
        )
        self.register_buffer(
            'target_initialized',
            torch.zeros(self.num_classes, dtype=torch.bool),
        )
        self.register_buffer(
            'target_counts',
            torch.zeros(self.num_classes, dtype=torch.long),
        )

    @torch.no_grad()
    def accumulate_source(self, features, labels):
        features = F.normalize(features.detach(), dim=1)
        labels = labels.detach().long()
        for c in range(self.num_classes):
            class_mask = labels.eq(c)
            if class_mask.any():
                self.source_feature_sums[c].add_(features[class_mask].sum(dim=0))
                self.source_counts[c].add_(int(class_mask.sum().item()))

    @torch.no_grad()
    def finalize_source(self):
        for c in range(self.num_classes):
            if self.source_counts[c].item() > 0:
                proto = self.source_feature_sums[c]
                self.source_prototypes[c].copy_(
                    F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)
                )
                self.source_initialized[c] = True

    @torch.no_grad()
    def update_target(self, features, labels, reliable_mask):
        reliable_mask = reliable_mask.bool()
        if not reliable_mask.any():
            return

        features = F.normalize(features.detach(), dim=1)
        labels = labels.detach().long()

        for c in range(self.num_classes):
            class_mask = reliable_mask & labels.eq(c)
            if not class_mask.any():
                continue

            batch_proto = F.normalize(
                features[class_mask].mean(dim=0, keepdim=True), dim=1
            ).squeeze(0)

            if self.target_initialized[c]:
                updated = (
                    self.target_prototypes[c] * self.momentum
                    + batch_proto * (1.0 - self.momentum)
                )
                self.target_prototypes[c].copy_(
                    F.normalize(updated.unsqueeze(0), dim=1).squeeze(0)
                )
            else:
                self.target_prototypes[c].copy_(batch_proto)
                self.target_initialized[c] = True

            self.target_counts[c].add_(int(class_mask.sum().item()))

    @torch.no_grad()
    def blended_prototypes(self):
        prototypes = self.source_prototypes.clone()
        for c in range(self.num_classes):
            source_ready = bool(self.source_initialized[c].item())
            target_ready = bool(self.target_initialized[c].item())

            if source_ready and target_ready:
                blended = (
                    self.source_anchor * self.source_prototypes[c]
                    + (1.0 - self.source_anchor) * self.target_prototypes[c]
                )
                prototypes[c] = F.normalize(
                    blended.unsqueeze(0), dim=1
                ).squeeze(0)
            elif target_ready:
                prototypes[c] = self.target_prototypes[c]

        return F.normalize(prototypes, dim=1)

    def consistency_loss(self, features, labels, reliable_mask,
                         temperature=0.2):
        """Class-balanced prototype contrastive loss on reliable target samples."""
        reliable_mask = reliable_mask.bool()
        if not reliable_mask.any():
            return features.sum() * 0.0

        if temperature <= 0.0:
            raise ValueError('prototype temperature must be positive')

        selected_features = F.normalize(features[reliable_mask], dim=1)
        selected_labels = labels[reliable_mask].long()
        prototypes = self.blended_prototypes().detach()
        logits = selected_features.mm(prototypes.t()) / float(temperature)
        per_sample = F.cross_entropy(logits, selected_labels, reduction='none')

        # Equalize classes inside the prototype regularizer so dominant pseudo
        # classes do not overwhelm minority expression classes.
        per_class = []
        for c in selected_labels.unique(sorted=True):
            class_mask = selected_labels.eq(c)
            per_class.append(per_sample[class_mask].mean())
        return torch.stack(per_class).mean()

    @torch.no_grad()
    def agreement_stats(self, features, labels, reliable_mask):
        reliable_mask = reliable_mask.bool()
        if not reliable_mask.any():
            return 0, 0, 0.0

        selected_features = F.normalize(features.detach()[reliable_mask], dim=1)
        selected_labels = labels.detach()[reliable_mask].long()
        prototypes = self.blended_prototypes()
        similarities = selected_features.mm(prototypes.t())
        prototype_predictions = similarities.argmax(dim=1)
        agreement = prototype_predictions.eq(selected_labels)
        assigned_similarity = similarities.gather(
            1, selected_labels.unsqueeze(1)
        ).squeeze(1)
        return (
            int(agreement.sum().item()),
            int(selected_labels.numel()),
            float(assigned_similarity.mean().item()),
        )


def prototype_weight_for_epoch(epoch, max_weight, warmup_epochs, ramp_epochs):
    """Warm up the target prototype bank before enabling its gradient loss."""
    if max_weight <= 0.0 or epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_weight)
    progress = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    return float(max_weight) * progress
