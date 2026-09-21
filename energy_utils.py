import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassDistributionBank(nn.Module):
    #
    #   源域类别分布库，用于判断目标域伪标签是否得到源域特征分布的支持。
    #   对每个类别 c，根据归一化后的源域特征估计：
    #       1. 类别均值 p_c
    #       2. 协方差 Sigma_c
    #       3. 精度矩阵 Sigma_c^{-1}
    #       4. 用于局部密度估计的 representative source features
    #       5. 每个类别自己的 log-energy 阈值
    #   对目标特征 z，如果 teacher 预测其类别为 c，则计算：
    #       D_c(z) = (z - p_c)^T Sigma_c^{-1} (z - p_c)
    #   其中 D_c(z) 是平方 Mahalanobis distance，用于衡量目标特征相对
    #   source 类中心的全局偏离程度。
    #   同时使用 Gaussian kernel 在相同 Mahalanobis 几何下估计局部密度：
    #       L_c(z)
    #   最终定义：
    #       log E_c(z)
    #           = log(D_c(z) + eps) - log L_c(z)
    #   它与：
    #       E_c(z) = D_c(z) / L_c(z)
    #   在排序意义上等价，但在高维空间使用 log-domain 能避免 KDE 密度过小
    #   导致的 numerical underflow。
    #   log-energy 越小，表示：
    #       - 距离 source 类中心较近；
    #       - 并且附近存在较强的 source 局部密度支持。
    #   每个类别的 threshold 使用 source representative 样本自身的
    #   leave-one-out energy 分布按 quantile 自动标定。
    #   注意：
    #       - mean 和 covariance 使用该类别所有 source features；
    #       - KDE 最多保留 max_density_samples 个 representative；
    #       - bandwidth 会乘 sqrt(feature_dim)，适配 512-D 等高维特征空间；
    #       - 本模块不参与反向传播，只负责统计和伪标签筛选。


    def __init__(self, num_classes, feature_dim, bandwidth=1.0,
                 covariance_shrinkage=0.05, energy_quantile=0.95,
                 max_density_samples=256, eps=1e-6):
        super().__init__()

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        # KDE 基础带宽
        self.bandwidth = float(bandwidth)
        # 协方差 shrinkage 强度
        self.covariance_shrinkage = float(covariance_shrinkage)
        # 用 source energy 分布的哪个 quantile 作为阈值。
        # quantile 越低，energy gate 越严格。
        self.energy_quantile = float(energy_quantile)
        # 每类最多保留多少个 source representative 用于 KDE
        self.max_density_samples = int(max_density_samples)

        # 数值稳定常数
        self.eps = float(eps)

        # ------------------------------------------------------------------
        # register_buffer:
        #
        # 这些 tensor 不是模型的可学习参数，不会被 optimizer 更新；
        # 但调用 .cuda() / .to(device) 时会跟随整个 module 自动迁移。
        # ------------------------------------------------------------------

        # 每一个类别的 source feature 均值 p_c。
        self.register_buffer(
            'source_means',
            torch.zeros(self.num_classes, self.feature_dim),
        )
        # 每个类别的 precision matrix，即 covariance inverse：
        self.register_buffer(
            'source_precisions',
            torch.zeros(self.num_classes, self.feature_dim, self.feature_dim),
        )
        # 每个类别参与 source distribution 建模的样本数量。
        self.register_buffer(
            'source_counts',
            torch.zeros(self.num_classes, dtype=torch.long),
        )
        # 每个类别自己的 log-energy threshold。
        self.register_buffer(
            'log_energy_thresholds',
            torch.full((self.num_classes,), float('inf')),
        )
        # 标记每个类别的 source distribution 是否成功建立
        self.register_buffer(
            'source_initialized',
            torch.zeros(self.num_classes, dtype=torch.bool),
        )
        # 每个类别临时保存所有 source feature batch。
        # finalize_source() 完成后会释放。
        self._source_chunks = [[] for _ in range(self.num_classes)]
        # 每个类别用于 KDE 的 representative features
        self._density_features = [None for _ in range(self.num_classes)]
        self._density_projected = [None for _ in range(self.num_classes)]
        # 缓存每个 representative 的： r^T P r
        self._density_quadratic = [None for _ in range(self.num_classes)]

    @torch.no_grad()

    def reset_source(self):

        # 清空已有source、distribution
        #
        # train.py 中会根据当前 EMA teacher 定期重新建立source distribution，因为随着teacher更新，feature space 也会发生变化。
        #
        self.source_means.zero_()
        self.source_precisions.zero_()
        self.source_counts.zero_()
        self.log_energy_thresholds.fill_(float('inf'))
        self.source_initialized.zero_()
        self._source_chunks = [[] for _ in range(self.num_classes)]
        self._density_features = [None for _ in range(self.num_classes)]
        self._density_projected = [None for _ in range(self.num_classes)]
        self._density_quadratic = [None for _ in range(self.num_classes)]

    @torch.no_grad()

    # 把源域每个batch的特征按类别缓存起来
    def accumulate_source(self, features, labels):
        features = F.normalize(features.detach(), dim=1)
        labels = labels.detach().long()
        for c in range(self.num_classes):
            class_mask = labels.eq(c)
            if class_mask.any():
                self._source_chunks[c].append(features[class_mask].clone())

    @torch.no_grad()

    # 用累积的源域特征，为每个类别计算均值、协方差（及其逆 / 精度矩阵）、密度代表点、以及能量阈值，然后释放原始特征缓存
    def finalize_source(self):
        identity = torch.eye(
            self.feature_dim,
            device=self.source_means.device,
            dtype=self.source_means.dtype,
        )

        for c in range(self.num_classes):
            if not self._source_chunks[c]:
                continue

            features = torch.cat(self._source_chunks[c], dim=0)
            count = int(features.size(0))
            self.source_counts[c] = count
            if count < 2:
                continue

            mean = features.mean(dim=0)
            centered = features - mean.unsqueeze(0)
            covariance = centered.t().mm(centered) / float(max(count - 1, 1))

            scale = torch.trace(covariance) / float(self.feature_dim)
            scale = scale.clamp_min(self.eps)
            covariance = (
                (1.0 - self.covariance_shrinkage) * covariance
                + self.covariance_shrinkage * scale * identity
                + self.eps * identity
            )

            # The covariance is symmetric positive definite after shrinkage and
            # jitter. Eigh is more stable here than a direct matrix inverse.
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            eigenvalues = eigenvalues.clamp_min(self.eps)
            precision = (
                eigenvectors * eigenvalues.reciprocal().unsqueeze(0)
            ).mm(eigenvectors.t())

            self.source_means[c].copy_(mean)
            self.source_precisions[c].copy_(precision)

            memory_size = min(count, self.max_density_samples)
            if memory_size == count:
                representatives = features
            else:
                indices = torch.linspace(
                    0,
                    count - 1,
                    steps=memory_size,
                    device=features.device,
                ).round().long()
                representatives = features.index_select(0, indices)

            representatives = representatives.contiguous()
            projected = representatives.mm(precision)
            quadratic = (projected * representatives).sum(dim=1)
            self._density_features[c] = representatives
            self._density_projected[c] = projected
            self._density_quadratic[c] = quadratic

            threshold = self._calibrate_class_threshold(c)
            self.log_energy_thresholds[c] = threshold
            self.source_initialized[c] = torch.isfinite(threshold)

        # Release full source feature chunks after statistics are finalized.
        self._source_chunks = [[] for _ in range(self.num_classes)]

    def _effective_bandwidth(self):
        return self.bandwidth * math.sqrt(float(self.feature_dim))

    @torch.no_grad()
    #计算其特征到第 class_index 类源均值的马氏距离平方，用源类别的协方差结构来衡量偏离程度
    def _mahalanobis_to_mean(self, features, class_index):
        centered = features - self.source_means[class_index].unsqueeze(0)
        projected = centered.mm(self.source_precisions[class_index])
        return (projected * centered).sum(dim=1).clamp_min(0.0)

    @torch.no_grad()

    # 马氏距离（Mahalanobis distance）的批量计算，用于衡量一批特征向量和某个类别下若干“代表点之间的距离
    def _pairwise_mahalanobis(self, features, class_index):
        representatives = self._density_features[class_index]
        if representatives is None:
            return features.new_empty((features.size(0), 0))

        precision = self.source_precisions[class_index]
        feature_projected = features.mm(precision)
        feature_quadratic = (feature_projected * features).sum(dim=1)
        representative_quadratic = self._density_quadratic[class_index]
        cross = feature_projected.mm(representatives.t())
        distances = (
            feature_quadratic.unsqueeze(1)
            + representative_quadratic.unsqueeze(0)
            - 2.0 * cross
        )
        return distances.clamp_min(0.0)

    @torch.no_grad()

    # 从成对马氏距离出发，计算每个样本的对数核密度（log kernel
    # density）估计。先处理空记忆库的边界情况，再计算高斯核的对数形式

    def _log_density(self, pairwise_distances, exclude_diagonal=False):
        memory_size = int(pairwise_distances.size(1))
        if memory_size == 0:
            return pairwise_distances.new_full(
                (pairwise_distances.size(0),), float('-inf')
            )

        h = self._effective_bandwidth()
        log_kernel = -pairwise_distances / (2.0 * h * h)

        if exclude_diagonal:
            if pairwise_distances.size(0) != memory_size or memory_size < 2:
                return pairwise_distances.new_full(
                    (pairwise_distances.size(0),), float('-inf')
                )
            diagonal = torch.eye(
                memory_size,
                device=pairwise_distances.device,
                dtype=torch.bool,
            )
            log_kernel = log_kernel.masked_fill(diagonal, float('-inf'))
            normalizer = float(memory_size - 1)
        else:
            normalizer = float(memory_size)

        return torch.logsumexp(log_kernel, dim=1) - math.log(normalizer)

    @torch.no_grad()

    # 为类别校准能量阈值的函数
    def _calibrate_class_threshold(self, class_index):
        representatives = self._density_features[class_index]
        if representatives is None or representatives.size(0) < 2:
            return self.source_means.new_tensor(float('inf'))

        global_deviation = self._mahalanobis_to_mean(
            representatives, class_index
        )
        pairwise = self._pairwise_mahalanobis(representatives, class_index)
        log_density = self._log_density(pairwise, exclude_diagonal=True)
        log_energy = torch.log(global_deviation + self.eps) - log_density
        finite = log_energy[torch.isfinite(log_energy)]
        if finite.numel() == 0:
            return self.source_means.new_tensor(float('inf'))
        return torch.quantile(finite, self.energy_quantile)

    @torch.no_grad()
    def gate(self, features, labels, candidate_mask):
        # 返回每个样本的对数能量以及一个源分布通过掩码。
        # candidate_mask 通常应当传入已有的双视角置信度掩码。对于没有有效源分布的类别，
        # 采取“失败开放”（fail open）策略，而不是把该类的所有伪标签全部删除。

        # 对输入特征做 L2 归一化，并切断梯度传播。
        # energy 模块只用于伪标签可靠性判断，不参与反向传播。
        features = F.normalize(features.detach(), dim=1)
        labels = labels.detach().long()

        # candidate_mask 表示当前允许进行 energy 判断的候选样本。
        # 在 v3 中通常是两个弱增强视图预测一致的样本。
        candidate_mask = candidate_mask.bool()

        log_energy = features.new_full((features.size(0),), float('nan'))

        # 默认先继承 candidate_mask。
        # 后续对于已经建立源域类别分布的类别，
        # 再根据 energy 阈值更新该类别样本是否通过分布验证
        pass_mask = candidate_mask.clone()

        # 对每个表情类别分别进行类别分布能量计算
        for c in range(self.num_classes):
            class_mask = candidate_mask & labels.eq(c)
            # 当前 batch 中没有类别 c 的候选样本，则跳过
            if not class_mask.any():
                continue

            # 如果类别 c 的源域分布模型尚未成功初始化
            # 就无法使用该类别的均值、协方差和 density 信息进行判断
            if not bool(self.source_initialized[c].item()):
                continue

            # 取出当前伪标签类别为 c 的目标域特征
            class_features = features[class_mask]
            # 1. 计算全局分布偏离程度
            global_deviation = self._mahalanobis_to_mean(class_features, c)
            # 2. 计算局部类别分布支持
            pairwise = self._pairwise_mahalanobis(class_features, c)
            # 计算类别 c 对当前目标样本的局部 log-density：
            log_density = self._log_density(pairwise, exclude_diagonal=False)
            # 3. 构造类别分布 energy
            class_log_energy = (
                torch.log(global_deviation + self.eps) - log_density
            )
            # 将类别 c 的 energy 写回整个 batch 对应的位置。
            log_energy[class_mask] = class_log_energy
            # 将目标样本 energy 与类别 c 的源域 energy 阈值进行比较。
            pass_mask[class_mask] = (
                class_log_energy <= self.log_energy_thresholds[c]
            )

        return log_energy, pass_mask
