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


def align_dual_view_probabilities(probs1, probs2, prior, power=0.5, ratio_max=3.0):
    """Use identical per-view correction for global statistics and batch labels."""
    return (distribution_align(probs1, prior, power, ratio_max),
            distribution_align(probs2, prior, power, ratio_max))


def weighted_mean_loss(losses, weights):
    """Normalize by effective weight; an empty mask has zero loss and gradient."""
    weights = weights.detach().to(losses).clamp_min(0.0)
    selected = torch.where(weights > 0, losses, torch.zeros_like(losses))
    return (selected * weights).sum() / weights.sum().clamp_min(1e-6)


def select_dual_view_pseudo_labels(probs1, probs2, thresholds, prior, args):
    """Select corrected predictions, including a strict two-view fallback."""
    conf1, pred1 = probs1.max(dim=1)
    conf2, pred2 = probs2.max(dim=1)
    confidence, labels = ((probs1 + probs2) * 0.5).max(dim=1)
    agreement = pred1.eq(pred2) & pred1.eq(labels)
    min_conf = torch.minimum(conf1, conf2)
    reliable = (agreement & (confidence >= thresholds.to(probs1)[labels])
                & (min_conf >= args.consistency_min_conf))
    if not torch.any(reliable):
        # Do not let one very confident view hide an uncertain second view.
        fallback = agreement & (min_conf >= max(args.fallback_conf,
                                                args.consistency_min_conf))
        if torch.any(fallback):
            indices = fallback.nonzero(as_tuple=False).squeeze(1)
            best = confidence[indices].argmax()
            reliable[indices[best]] = True

    prior = prior.to(probs1).clamp_min(1e-6)
    balance = ((1.0 / prior.numel()) / prior).pow(args.distribution_power * 0.5)
    balance = balance.clamp(1.0 / args.distribution_ratio_max,
                            args.distribution_ratio_max)
    weights = (confidence * min_conf).clamp_min(0.0).sqrt() * balance[labels]
    weights = weights.clamp(0.0, args.pseudo_weight_max) * reliable.float()
    return labels, reliable.float(), weights, int(agreement.sum().item())


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
            finite = torch.isfinite(class_features).all(dim=1)
            finite &= class_features.norm(dim=1) > 1e-6
            class_features = class_features[finite]
            if class_features.size(0) == 0:
                continue
            if sample_weights is None:
                center = class_features.mean(dim=0)
            else:
                weights = sample_weights[mask][finite].clamp_min(0.0)
                if not torch.isfinite(weights).all() or weights.sum() <= 1e-6:
                    continue
                denom = weights.sum()
                center = (class_features * weights.unsqueeze(1)).sum(dim=0) / denom

            if center.norm() <= 1e-6:
                continue
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
        # Prototypes are fixed anchors; BOTH terms must depend on live student
        # features. Comparing memory prototypes to each other has no gradient.
        similarity = features_valid.mm(self.prototypes.detach().t())
        positive = similarity.gather(1, labels_valid.unsqueeze(1)).squeeze(1)
        compact = (1.0 - positive).clamp_min(0.0)
        negatives = self.initialized.unsqueeze(0).expand_as(similarity).clone()
        negatives.scatter_(1, labels_valid.unsqueeze(1), False)
        separation = F.relu(similarity - positive.unsqueeze(1) + self.margin)
        separation = (separation * negatives).sum(dim=1) / negatives.sum(dim=1).clamp_min(1)
        per_sample = compact + separation
        total = (weighted_mean_loss(per_sample, sample_weights[valid])
                 if sample_weights is not None else per_sample.mean())
        return torch.nan_to_num(total, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

    def state_dict(self):
        return {
            'prototypes': self.prototypes.detach().cpu(),
            'initialized': self.initialized.detach().cpu(),
        }

    def load_state_dict(self, state):
        self.prototypes.copy_(state['prototypes'].to(self.device))
        self.initialized.copy_(state['initialized'].to(self.device))
