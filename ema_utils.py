import copy
import torch
import torch.nn.functional as F


def create_ema_teacher(student):
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher, student, decay, global_step):
    """EMA update with a short bias-correction warm start."""
    dynamic_decay = min(float(decay), 1.0 - 1.0 / float(global_step + 1))

    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher_params.items():
        teacher_param.mul_(dynamic_decay).add_(
            student_params[name].detach(), alpha=1.0 - dynamic_decay
        )

    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher_buffers.items():
        source = student_buffers[name].detach()
        if teacher_buffer.dtype.is_floating_point:
            teacher_buffer.mul_(dynamic_decay).add_(source, alpha=1.0 - dynamic_decay)
        else:
            teacher_buffer.copy_(source)

    teacher.eval()


def distribution_align(probs, prior, power=0.5, ratio_max=3.0):
    """Mild inverse-prior correction for target class imbalance."""
    prior = prior.to(device=probs.device, dtype=probs.dtype).clamp_min(1e-6)
    uniform = torch.full_like(prior, 1.0 / float(prior.numel()))
    ratio = torch.pow(uniform / prior, power)
    ratio = torch.clamp(ratio, min=1.0 / ratio_max, max=ratio_max)
    corrected = probs * ratio.unsqueeze(0)
    return corrected / corrected.sum(dim=1, keepdim=True).clamp_min(1e-12)


class PrototypeMemory:
    """EMA class prototypes used by the improved target affinity loss."""

    def __init__(self, num_classes, feature_dim, device,
                 momentum=0.9, margin=0.2):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.momentum = momentum
        self.margin = margin
        self.prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, features, labels, sample_weights=None):
        if features.numel() == 0:
            return

        features = F.normalize(features.detach(), dim=1)
        labels = labels.detach()
        if sample_weights is not None:
            sample_weights = sample_weights.detach().to(features.device)

        for c in range(self.num_classes):
            mask = labels == c
            if not torch.any(mask):
                continue

            class_features = features[mask]
            if sample_weights is None:
                center = class_features.mean(dim=0)
            else:
                weights = sample_weights[mask].clamp_min(0.0)
                denom = weights.sum().clamp_min(1e-6)
                center = (class_features * weights.unsqueeze(1)).sum(dim=0) / denom

            center = F.normalize(center.unsqueeze(0), dim=1).squeeze(0)
            if self.initialized[c]:
                center = (
                    self.momentum * self.prototypes[c]
                    + (1.0 - self.momentum) * center
                )
                center = F.normalize(center.unsqueeze(0), dim=1).squeeze(0)

            self.prototypes[c].copy_(center)
            self.initialized[c] = True

    def loss(self, features, labels, sample_weights=None):
        if features.numel() == 0:
            return features.sum() * 0.0

        valid = self.initialized.index_select(0, labels)
        if not torch.any(valid):
            return features.sum() * 0.0

        features_valid = F.normalize(features[valid], dim=1)
        labels_valid = labels[valid]
        prototypes_valid = self.prototypes.index_select(0, labels_valid).detach()
        compact_loss = 1.0 - torch.sum(features_valid * prototypes_valid, dim=1)

        if sample_weights is not None:
            weights = sample_weights[valid].clamp_min(0.0)
            compact_loss = (compact_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            compact_loss = compact_loss.mean()

        initialized_prototypes = self.prototypes[self.initialized]
        if initialized_prototypes.size(0) >= 2:
            initialized_prototypes = F.normalize(initialized_prototypes, dim=1)
            cosine = initialized_prototypes.mm(initialized_prototypes.t())
            off_diagonal = ~torch.eye(
                cosine.size(0), dtype=torch.bool, device=cosine.device
            )
            separation_loss = F.relu(cosine[off_diagonal] - self.margin).mean()
        else:
            separation_loss = compact_loss.new_tensor(0.0)

        total = compact_loss + separation_loss
        return torch.nan_to_num(total, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

    def state_dict(self):
        return {
            'prototypes': self.prototypes.detach().cpu(),
            'initialized': self.initialized.detach().cpu(),
        }

    def load_state_dict(self, state):
        self.prototypes.copy_(state['prototypes'].to(self.device))
        self.initialized.copy_(state['initialized'].to(self.device))
