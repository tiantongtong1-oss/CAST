from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import math
import numpy as np
from sklearn import svm
import itertools
from sklearn.neighbors import KernelDensity

def cal_weight(x):
    # Absent classes must not receive exp(1 / 1e-6), which overflows
    # and suppresses every present class.
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x) & (np.abs(x) > 1e-12)
    weights = np.zeros_like(x)
    if valid.any():
        scores = 1.0 / x[valid]
        scores -= scores.max()
        weights[valid] = np.exp(scores) / np.exp(scores).sum()
    return weights

def mmd_loss(source_features, target_features):
    num_samples = min(source_features.size(0), target_features.size(0))
    if num_samples == 0:
        return (source_features.sum() + target_features.sum()) * 0.0
    matched_source = source_features[torch.randperm(source_features.size(0), device=source_features.device)[:num_samples]]
    loss = 0.
    for _ in range(6):
        matched_target = target_features[torch.randperm(target_features.size(0), device=target_features.device)[:num_samples]]

        kernels = compute_kernel_matrix(matched_source, matched_target)
        XX = kernels[:num_samples, :num_samples]
        YY = kernels[num_samples:, num_samples:]
        XY = kernels[:num_samples, num_samples:]
        YX = kernels[num_samples:, :num_samples]
        loss += torch.mean(XX + YY - XY -YX)
    return loss / 6

def compute_kernel_matrix(source, target, kernel_mul=2.0, kernel_num=5):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    # bandwidth=5
    bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples) ##bandwidth=0.5
    bandwidth = bandwidth.clamp_min(torch.finfo(total.dtype).eps)
    bandwidth /= (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)#/len(kernel_val)

def remove_element(matrix, index):
    remaining = [part for i, part in enumerate(matrix) if i != index and part.shape[0]]
    return torch.cat(remaining, dim=0) if remaining else matrix[index].new_empty((0, matrix[index].shape[1]))


class Model(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=7, pretrained=True, drop_rate=0.5):
        super(Model, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(num_classes)
        self.kde = KernelDensity(bandwidth=0.2, kernel='gaussian')

        if backbone == 'resnet18':
            self.feature = nn.Sequential(*list(models.resnet18(pretrained).children())[:-1], nn.Flatten(), nn.Dropout(drop_rate))
            self.fc = nn.Linear(512, num_classes, bias = False)

        elif backbone == 'resnet50':
            self.feature = nn.Sequential(*list(models.resnet50(pretrained).children())[:-1], nn.Flatten(),  nn.Dropout(drop_rate), nn.Linear(2048, 512))
            self.fc = nn.Linear(512, num_classes, bias = False)

        elif backbone == 'mobilenet_v2':
            self.feature = nn.Sequential(*list(models.mobilenet_v2(pretrained).children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop_rate), nn.Linear(1280, 512), nn.Dropout(drop_rate)) #62720  81920
            self.fc = nn.Linear(512, num_classes, bias = False)

        else:
            raise ValueError('Backbone Error!')

    def forward(self, x, targets, idx, mode='train', task='target', epoch=0,
                source_count=None, compute_affinity=True):
        fea = self.feature(x)
        out = self.bn(self.fc(fea))
        batch = fea.shape[0]
        if mode == 'train':
              if not compute_affinity:
                    return [out, fea.new_zeros(())]
              if task == 'source':
                    features = self.split_feature_makeLD(fea, targets)
                    weight = self.volume(features)
                    w = fea.new_tensor(weight)
                    inter_loss = fea.new_zeros(())
                    for i in range(self.num_classes):
                        fea = features[i]
                        if len(fea) != 0:
                            fea_= remove_element(features, i)
                            inter_loss += mmd_loss(fea, fea_) * w[i]
                    affinity_loss = -1 * inter_loss
                    return [out, affinity_loss]

              if task == 'target':  ####calculate affinity loss for all source and confident target samples.
                    # Split domains BEFORE filtering: unequal source/target batches
                    # and sparse masks must not move target samples into source.
                    n_source = batch // 2 if source_count is None else source_count
                    mask = idx.bool()
                    source_features = self.split_feature_makeLD(
                        fea[:n_source][mask[:n_source]], targets[:n_source][mask[:n_source]])
                    target_features = self.split_feature_makeLD(
                        fea[n_source:][mask[n_source:]], targets[n_source:][mask[n_source:]])
                    features = self.split_feature_makeLD(fea[mask], targets[mask])
                    weight = self.volume(features)
                    w = fea.new_tensor(weight)
                    intra_loss, inter_loss = fea.new_zeros(()), fea.new_zeros(())
                    for i in range(self.num_classes):
                        fea_s = source_features[i]
                        fea_t = target_features[i]
                        if len(fea_s) != 0 and len(fea_t) != 0:
                            intra_loss += mmd_loss(fea_s, fea_t) * w[i]

                        fea = features[i]
                        if len(fea) != 0:
                            fea_= remove_element(features, i)
                            inter_loss += mmd_loss(fea, fea_) * w[i]

                    affinity_loss = intra_loss - inter_loss
                    return [out, affinity_loss]

        else:
          return out, fea.cpu()


    def split_feature_makeLD(self, x, target):
        x_parts = []
        inds = []
        for c in range(self.num_classes):
            ind = (target == c).nonzero()
            inds.append(ind)
            x_parts.append(x[ind[:, 0], :])
        return x_parts

    def volume(self, features):
        volume = np.zeros(shape = (len(features)))
        for idx in range(len(features)):
            f = features[idx].cpu().detach().numpy()
            if len(f) != 0:
                self.kde.fit(f)
                v = np.sum(1/(self.kde.score_samples(f)))
            else:
                v = 0.0
            volume[idx] = v
        weight = cal_weight(volume)
        return weight

