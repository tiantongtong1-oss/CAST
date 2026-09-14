from torch import nn
import torch
from torchvision import models


def build_backbone(name, pretrained):
    """Keep legacy ImageNet V1 initialization and support both torchvision APIs."""
    enums = {'mobilenet_v2': 'MobileNet_V2_Weights',
             'resnet18': 'ResNet18_Weights', 'resnet50': 'ResNet50_Weights'}
    builder = getattr(models, name)
    weights_enum = getattr(models, enums[name], None)
    if weights_enum is not None:
        return builder(weights=weights_enum.IMAGENET1K_V1 if pretrained else None)
    return builder(pretrained=pretrained)


def compute_kernel_matrix(source, target, kernel_mul=2.0, kernel_num=5):
    """Multi-kernel Gaussian matrix used by MK-MMD."""
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    # Materialize only N x N distances, not an N x N x feature_dim tensor.
    l2_distance = torch.cdist(total, total, p=2).pow(2)

    if n_samples <= 1:
        return torch.zeros_like(l2_distance)

    bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)
    bandwidth = torch.clamp(bandwidth, min=1e-12)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_loss(source_features, target_features):
    """Unbiased MK-MMD estimator."""
    ns = source_features.size(0)
    nt = target_features.size(0)
    if ns < 2 or nt < 2:
        return (source_features.sum() + target_features.sum()) * 0.0

    kernels = compute_kernel_matrix(source_features, target_features)
    xx = kernels[:ns, :ns]
    yy = kernels[ns:, ns:]
    xy = kernels[:ns, ns:]

    xx_term = (xx.sum() - torch.diagonal(xx).sum()) / (ns * (ns - 1))
    yy_term = (yy.sum() - torch.diagonal(yy).sum()) / (nt * (nt - 1))
    xy_term = xy.mean()
    return xx_term + yy_term - 2.0 * xy_term


def remove_element(features, index):
    others = [feature for i, feature in enumerate(features)
              if i != index and feature.size(0) > 0]
    if not others:
        return features[index].new_empty((0, features[index].size(1)))
    return torch.cat(others, dim=0)


class Model(nn.Module):
    """Shared FER backbone plus CAST DDRL/CCDR feature constraints.

    The improved branch keeps the original MK-MMD based DDRL and exposes
    features to the training loop so the target prototype affinity loss can
    be computed with reliable EMA-teacher pseudo labels.
    """

    def __init__(self, backbone='mobilenet_v2', num_classes=7, pretrained=True,
                 drop_rate=0.5, density_bandwidth=0.2,
                 class_weight_max=1.0, loss_clip=5.0):
        super(Model, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.density_bandwidth = density_bandwidth
        self.class_weight_max = class_weight_max
        self.loss_clip = loss_clip
        self.bn = nn.BatchNorm1d(num_classes)

        if backbone == 'resnet18':
            self.feature = nn.Sequential(
                *list(build_backbone('resnet18', pretrained).children())[:-1],
                nn.Flatten(),
                nn.Dropout(drop_rate)
            )
            self.feature_dim = 512
            self.fc = nn.Linear(self.feature_dim, num_classes, bias=False)

        elif backbone == 'resnet50':
            self.feature = nn.Sequential(
                *list(build_backbone('resnet50', pretrained).children())[:-1],
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(2048, 512)
            )
            self.feature_dim = 512
            self.fc = nn.Linear(self.feature_dim, num_classes, bias=False)

        elif backbone == 'mobilenet_v2':
            self.feature = nn.Sequential(
                *list(build_backbone('mobilenet_v2', pretrained).children())[:-1],
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(1280, 512),
                nn.Dropout(drop_rate)
            )
            self.feature_dim = 512
            self.fc = nn.Linear(self.feature_dim, num_classes, bias=False)

        else:
            raise ValueError('Backbone Error!')

    def _stabilize_loss(self, loss):
        loss = torch.nan_to_num(
            loss,
            nan=0.0,
            posinf=self.loss_clip,
            neginf=-self.loss_clip,
        )
        return torch.clamp(loss, min=-self.loss_clip, max=self.loss_clip)

    def forward(self, x, targets=None, idx=None, mode='train', task='target',
                epoch=0, source_count=None):
        fea = self.feature(x)
        out = self.bn(self.fc(fea))

        if mode != 'train':
            # Teacher features stay on-device for prototype memory updates.
            return out, fea

        if task == 'source':
            features = self.split_feature_makeLD(fea, targets)
            eta = self.volume(features)
            inter_loss = fea.new_tensor(0.0)

            for i in range(self.num_classes):
                class_features = features[i]
                other_features = remove_element(features, i)
                if class_features.size(0) >= 2 and other_features.size(0) >= 2:
                    inter_loss += mmd_loss(class_features, other_features) * eta[i]

            class_separation_loss = -inter_loss / self.num_classes
            class_separation_loss = self._stabilize_loss(class_separation_loss)
            return [out, class_separation_loss, fea]

        if task == 'target':
            if idx is None or source_count is None:
                raise ValueError('Target training requires confidence mask idx and source_count.')

            source_count = int(source_count)
            if not 0 <= source_count <= fea.size(0):
                raise ValueError('source_count must partition the original batch.')
            source_mask = idx[:source_count] > 0
            target_mask = idx[source_count:] > 0
            source_fea = fea[:source_count][source_mask]
            source_targets = targets[:source_count][source_mask]
            target_fea = fea[source_count:][target_mask]
            target_targets = targets[source_count:][target_mask]
            if source_fea.size(0) + target_fea.size(0) == 0:
                zero = fea.sum() * 0.0
                return [out, zero, fea, zero, zero]

            fea_selected = torch.cat((source_fea, target_fea), dim=0)
            targets_selected = torch.cat((source_targets, target_targets), dim=0)

            source_features = self.split_feature_makeLD(source_fea, source_targets)
            target_features = self.split_feature_makeLD(target_fea, target_targets)
            all_features = self.split_feature_makeLD(fea_selected, targets_selected)
            eta = self.volume(all_features)

            intra_loss = fea.new_tensor(0.0)
            inter_loss = fea.new_tensor(0.0)
            aligned_classes = separated_classes = 0

            for i in range(self.num_classes):
                fea_s = source_features[i]
                fea_t = target_features[i]
                if fea_s.size(0) >= 2 and fea_t.size(0) >= 2:
                    intra_loss += mmd_loss(fea_s, fea_t) * eta[i]
                    aligned_classes += 1

                class_features = all_features[i]
                other_features = remove_element(all_features, i)
                if class_features.size(0) >= 2 and other_features.size(0) >= 2:
                    inter_loss += mmd_loss(class_features, other_features) * eta[i]
                    separated_classes += 1

            # Missing classes must not dilute the available constraints.
            domain_alignment_loss = intra_loss / max(aligned_classes, 1)
            class_enhancement_loss = -inter_loss / max(separated_classes, 1)

            domain_alignment_loss = self._stabilize_loss(domain_alignment_loss)
            class_enhancement_loss = self._stabilize_loss(class_enhancement_loss)
            ddrl_loss = domain_alignment_loss + class_enhancement_loss
            return [out, ddrl_loss, fea,
                    domain_alignment_loss, class_enhancement_loss]

        raise ValueError('Unknown task: %s' % task)

    def split_feature_makeLD(self, x, target):
        if x.size(0) == 0:
            return [x.new_empty((0, self.feature_dim)) for _ in range(self.num_classes)]

        x_parts = []
        for c in range(self.num_classes):
            ind = (target == c).nonzero(as_tuple=False).squeeze(1)
            x_parts.append(torch.index_select(x, 0, ind))
        return x_parts

    def volume(self, features):
        """Stable class-level representation weights used by CCDR.

        Empty classes are assigned zero weight. The upper bound prevents an
        unexpectedly sparse batch from producing a dominating class weight.
        """
        if not features:
            return []

        h = float(self.density_bandwidth)
        weights = []
        for feature in features:
            n = feature.size(0)
            if n == 0:
                weights.append(0.0)
                continue

            with torch.no_grad():
                f = feature.detach()
                distance_sq = torch.cdist(f, f, p=2).pow(2)
                gaussian_kernel = torch.exp(-distance_sq / (2.0 * h * h))
                rho = gaussian_kernel.sum(dim=1) / (float(n) * h)
                rho = rho.clamp_min(1e-12)
                volume = torch.sum(1.0 / rho)
                eta = 1.0 / volume.clamp_min(1e-12)
                eta = torch.nan_to_num(eta, nan=0.0, posinf=self.class_weight_max)
                eta = torch.clamp(eta, min=0.0, max=self.class_weight_max)
                weights.append(float(eta.item()))

        return weights
