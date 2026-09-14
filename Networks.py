from torch import nn
import torch
from torchvision import models
import numpy as np
from sklearn.neighbors import KernelDensity


def compute_kernel_matrix(source, target, kernel_mul=2.0, kernel_num=5):
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if n_samples <= 1:
        return torch.zeros_like(l2_distance)

    bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)
    bandwidth = torch.clamp(bandwidth, min=1e-12)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_loss(source_features, target_features):
    """Unbiased multi-kernel MMD matching Eq. (9)-(10) in the CAST paper."""
    ns = source_features.size(0)
    nt = target_features.size(0)
    if ns < 2 or nt < 2:
        return source_features.new_tensor(0.0)

    kernels = compute_kernel_matrix(source_features, target_features)
    xx = kernels[:ns, :ns]
    yy = kernels[ns:, ns:]
    xy = kernels[:ns, ns:]

    xx_term = (xx.sum() - torch.diagonal(xx).sum()) / (ns * (ns - 1))
    yy_term = (yy.sum() - torch.diagonal(yy).sum()) / (nt * (nt - 1))
    xy_term = xy.mean()
    return xx_term + yy_term - 2.0 * xy_term


def remove_element(features, index):
    others = [feature for i, feature in enumerate(features) if i != index and feature.size(0) > 0]
    if not others:
        return features[index].new_empty((0, features[index].size(1)))
    return torch.cat(others, dim=0)


class Model(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=7, pretrained=True, drop_rate=0.5):
        super(Model, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(num_classes)
        self.kde = KernelDensity(bandwidth=0.2, kernel='gaussian')

        if backbone == 'resnet18':
            self.feature = nn.Sequential(
                *list(models.resnet18(pretrained=pretrained).children())[:-1],
                nn.Flatten(),
                nn.Dropout(drop_rate)
            )
            self.fc = nn.Linear(512, num_classes, bias=False)

        elif backbone == 'resnet50':
            self.feature = nn.Sequential(
                *list(models.resnet50(pretrained=pretrained).children())[:-1],
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(2048, 512)
            )
            self.fc = nn.Linear(512, num_classes, bias=False)

        elif backbone == 'mobilenet_v2':
            self.feature = nn.Sequential(
                *list(models.mobilenet_v2(pretrained=pretrained).children())[:-1],
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(1280, 512),
                nn.Dropout(drop_rate)
            )
            self.fc = nn.Linear(512, num_classes, bias=False)

        else:
            raise ValueError('Backbone Error!')

    def forward(self, x, targets=None, idx=None, mode='train', task='target', epoch=0, source_count=None):
        fea = self.feature(x)
        out = self.bn(self.fc(fea))

        if mode != 'train':
            return out, fea.cpu()

        if task == 'source':
            features = self.split_feature_makeLD(fea, targets)
            eta = self.volume(features)
            inter_loss = fea.new_tensor(0.0)

            for i in range(self.num_classes):
                class_features = features[i]
                other_features = remove_element(features, i)
                if class_features.size(0) >= 2 and other_features.size(0) >= 2:
                    inter_loss += mmd_loss(class_features, other_features) * eta[i]

            # Eq. (11)-(12): enlarge inter-class discrepancy.
            affinity_loss = -inter_loss / self.num_classes
            return [out, affinity_loss]

        if task == 'target':
            if idx is None or source_count is None:
                raise ValueError('Target training requires confidence mask idx and source_count.')

            valid_idx = (idx == 1).nonzero(as_tuple=False).squeeze(1)
            fea_selected = torch.index_select(fea, 0, valid_idx)
            targets_selected = torch.index_select(targets, 0, valid_idx)

            # All source samples are placed first and are always marked confident.
            source_count = int(source_count)
            source_count = min(source_count, fea_selected.size(0))
            source_fea = fea_selected[:source_count]
            source_targets = targets_selected[:source_count]
            target_fea = fea_selected[source_count:]
            target_targets = targets_selected[source_count:]

            source_features = self.split_feature_makeLD(source_fea, source_targets)
            target_features = self.split_feature_makeLD(target_fea, target_targets)
            all_features = self.split_feature_makeLD(fea_selected, targets_selected)
            eta = self.volume(all_features)

            intra_loss = fea.new_tensor(0.0)
            inter_loss = fea.new_tensor(0.0)

            for i in range(self.num_classes):
                fea_s = source_features[i]
                fea_t = target_features[i]
                if fea_s.size(0) >= 2 and fea_t.size(0) >= 2:
                    intra_loss += mmd_loss(fea_s, fea_t) * eta[i]

                class_features = all_features[i]
                other_features = remove_element(all_features, i)
                if class_features.size(0) >= 2 and other_features.size(0) >= 2:
                    inter_loss += mmd_loss(class_features, other_features) * eta[i]

            # Eq. (12): class-conditional alignment plus inter-class separation.
            affinity_loss = (intra_loss - inter_loss) / self.num_classes
            return [out, affinity_loss]

        raise ValueError('Unknown task: %s' % task)

    def split_feature_makeLD(self, x, target):
        x_parts = []
        if x.size(0) == 0:
            feature_dim = self.fc.in_features
            return [x.new_empty((0, feature_dim)) for _ in range(self.num_classes)]

        for c in range(self.num_classes):
            ind = (target == c).nonzero(as_tuple=False).squeeze(1)
            x_parts.append(torch.index_select(x, 0, ind))
        return x_parts

    def volume(self, features):
        """CCDR class-level representation modulation, Eq. (2)-(5).

        sklearn KernelDensity.score_samples returns log-density, so convert it
        back to density before computing V_c = sum_i 1/rho_i and eta_c = 1/V_c.
        """
        eta = np.zeros(len(features), dtype=np.float64)
        for idx, feature in enumerate(features):
            f = feature.detach().cpu().numpy()
            if len(f) == 0:
                eta[idx] = 0.0
                continue

            self.kde.fit(f)
            log_density = self.kde.score_samples(f)
            density = np.exp(log_density)
            density = np.maximum(density, 1e-12)
            volume = np.sum(1.0 / density)
            eta[idx] = 1.0 / max(volume, 1e-12)

        return eta
