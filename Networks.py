import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from torch import nn
from torchvision import models


def mmd_loss(source_features, target_features):
    """Multi-kernel MMD used by CAST DDRL."""
    num_samples = min(source_features.size(0), target_features.size(0))
    if num_samples == 0:
        return (source_features.sum() + target_features.sum()) * 0.0

    device = source_features.device
    matched_source = source_features[
        torch.randperm(source_features.size(0), device=device)[:num_samples]
    ]

    loss = source_features.new_zeros(())
    for _ in range(6):
        matched_target = target_features[
            torch.randperm(target_features.size(0), device=device)[:num_samples]
        ]
        kernels = compute_kernel_matrix(matched_source, matched_target)
        xx = kernels[:num_samples, :num_samples]
        yy = kernels[num_samples:, num_samples:]
        xy = kernels[:num_samples, num_samples:]
        yx = kernels[num_samples:, :num_samples]
        loss = loss + torch.mean(xx + yy - xy - yx)
    return loss / 6.0


def compute_kernel_matrix(source, target, kernel_mul=2.0, kernel_num=5):
    total = torch.cat([source, target], dim=0)
    n_samples = int(total.size(0))

    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)

    denominator = max(n_samples * n_samples - n_samples, 1)
    bandwidth = l2_distance.detach().sum() / float(denominator)
    bandwidth = bandwidth.clamp_min(torch.finfo(total.dtype).eps)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-l2_distance / bandwidth_temp.clamp_min(torch.finfo(total.dtype).eps))
        for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)


def remove_element(parts, index):
    remaining = [part for i, part in enumerate(parts) if i != index and part.shape[0] > 0]
    if remaining:
        return torch.cat(remaining, dim=0)
    return parts[index].new_empty((0, parts[index].shape[1]))


class Model(nn.Module):
    def __init__(self, backbone="resnet50", num_classes=7, pretrained=True, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(num_classes)
        self.kde = KernelDensity(bandwidth=0.2, kernel="gaussian")

        if backbone == "resnet18":
            self.feature = nn.Sequential(
                *list(models.resnet18(pretrained=pretrained).children())[:-1],
                nn.Flatten(),
                nn.Dropout(drop_rate),
            )
            self.fc = nn.Linear(512, num_classes, bias=False)

        elif backbone == "resnet50":
            self.feature = nn.Sequential(
                *list(models.resnet50(pretrained=pretrained).children())[:-1],
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(2048, 512),
            )
            self.fc = nn.Linear(512, num_classes, bias=False)

        elif backbone == "mobilenet_v2":
            self.feature = nn.Sequential(
                *list(models.mobilenet_v2(pretrained=pretrained).children())[:-1],
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(1280, 512),
                nn.Dropout(drop_rate),
            )
            self.fc = nn.Linear(512, num_classes, bias=False)
        else:
            raise ValueError("Backbone Error!")

    def forward(
        self,
        x,
        targets,
        idx,
        mode="train",
        task="target",
        epoch=0,
        source_count=None,
        compute_affinity=True,
    ):
        fea = self.feature(x)
        out = self.bn(self.fc(fea))

        if mode != "train":
            return out, fea.cpu()

        if not compute_affinity:
            return [out, fea.new_zeros(())]

        if task == "source":
            features = self.split_feature_makeLD(fea, targets)
            weight = fea.new_tensor(self.volume(features))
            inter_loss = fea.new_zeros(())

            for c in range(self.num_classes):
                fea_c = features[c]
                fea_other = remove_element(features, c)
                if fea_c.shape[0] > 0 and fea_other.shape[0] > 0:
                    inter_loss = inter_loss + mmd_loss(fea_c, fea_other) * weight[c]

            return [out, -inter_loss]

        if task == "target":
            if idx is None:
                raise ValueError("Target affinity requires a source/target confidence mask.")

            mask = idx.bool()
            n_source = fea.shape[0] // 2 if source_count is None else int(source_count)
            if mask.numel() != fea.shape[0]:
                raise ValueError("Affinity mask length must match the concatenated batch size.")

            source_mask = mask[:n_source]
            target_mask = mask[n_source:]

            source_fea = fea[:n_source][source_mask]
            source_targets = targets[:n_source][source_mask]
            target_fea = fea[n_source:][target_mask]
            target_targets = targets[n_source:][target_mask]

            source_features = self.split_feature_makeLD(source_fea, source_targets)
            target_features = self.split_feature_makeLD(target_fea, target_targets)
            all_features = self.split_feature_makeLD(fea[mask], targets[mask])

            weight = fea.new_tensor(self.volume(all_features))
            intra_loss = fea.new_zeros(())
            inter_loss = fea.new_zeros(())

            for c in range(self.num_classes):
                fea_s = source_features[c]
                fea_t = target_features[c]
                if fea_s.shape[0] > 0 and fea_t.shape[0] > 0:
                    intra_loss = intra_loss + mmd_loss(fea_s, fea_t) * weight[c]

                fea_c = all_features[c]
                fea_other = remove_element(all_features, c)
                if fea_c.shape[0] > 0 and fea_other.shape[0] > 0:
                    inter_loss = inter_loss + mmd_loss(fea_c, fea_other) * weight[c]

            return [out, intra_loss - inter_loss]

        raise ValueError("Unknown training task: %s" % task)

    def split_feature_makeLD(self, x, target):
        parts = []
        for c in range(self.num_classes):
            ind = (target == c).nonzero(as_tuple=False).flatten()
            parts.append(x.index_select(0, ind) if ind.numel() else x.new_empty((0, x.shape[1])))
        return parts

    def volume(self, features):
        """CCDR class-level representation weights.

        sklearn KernelDensity.score_samples returns log-density. CAST Eq. (3)
        needs V_c = sum_i 1 / rho_i, so we compute log(V_c) with log-sum-exp
        for numerical stability and then normalize eta_c = 1 / V_c over
        classes that are actually present in the batch.
        """
        log_eta = np.full(len(features), -np.inf, dtype=np.float64)

        for c, feature in enumerate(features):
            if feature.shape[0] == 0:
                continue

            f = feature.detach().cpu().numpy()
            self.kde.fit(f)
            log_rho = self.kde.score_samples(f)

            # log V_c = log sum_i exp(-log rho_i)
            z = -log_rho
            z_max = np.max(z)
            log_volume = z_max + np.log(np.exp(z - z_max).sum())
            log_eta[c] = -log_volume

        valid = np.isfinite(log_eta)
        weight = np.zeros(len(features), dtype=np.float64)
        if valid.any():
            scores = log_eta[valid]
            scores = scores - scores.max()
            scores = np.exp(scores)
            denom = scores.sum()
            if np.isfinite(denom) and denom > 0:
                weight[valid] = scores / denom
            else:
                weight[valid] = 1.0 / float(valid.sum())

        return weight
