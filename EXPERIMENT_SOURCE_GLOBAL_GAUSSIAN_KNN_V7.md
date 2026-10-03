# Source Global Gaussian KNN reliability v7

本分支实现“源域真标签高斯区域 + 全局方差动量 + kNN 密度/支持可靠性评分”。

## 统计模型

每次 reliability refresh 使用 EMA Teacher 扫描完整 RAF-DB source train 的确定性视图。源域真实标签用于分组；目标域真实标签完全不参与高斯拟合或可靠性评分。

特征先做 L2 normalize。对每个类别 c 计算源域类中心：

```text
mu_c = mean(normalize(z_i) | y_i = c)
```

所有类别共享一个 source-only isotropic within-class variance：

```text
observed_global_var
  = sum_c sum_i:y_i=c ||z_i - mu_c||^2
    / (N_source * feature_dim)
```

它等价于按样本数汇总所有类别的逐维 MLE 类内方差后，再对 feature dimension 求平均。

第一次 refresh 直接初始化；之后使用递增 momentum：

```text
beta_t = linear(start_momentum, end_momentum, refresh_progress)
global_var_t
  = beta_t * global_var_(t-1)
    + (1 - beta_t) * observed_global_var_t
sigma_t = sqrt(global_var_t)
```

默认从 0.70 线性增加到 0.95，共 30 次 refresh。

## 类高斯置信区间

每个类别使用自己的 source center，但共享 global variance：

```text
z | class=c ~ N(mu_c, global_var * I)
D_c(z) = ||normalize(z) - mu_c||^2 / global_var
G_c(z) = D_c(z) <= chi2_quantile(feature_dim, distribution_mass)
```

默认 `distribution_mass=0.95`。

候选伪标签样本必须首先落在它的预测类别 c 的 source Gaussian 区域内，否则 reliability score 为 0。

## kNN 邻居检查

目标训练集只缓存 EMA Teacher 的预测标签、置信度、特征和稳定 sample ID。目标真实标签被忽略。

对候选 i 在完整目标 memory 中搜索 k 个最近邻。每个邻居 j 只有同时满足以下条件才成为 support：

1. teacher predicted class 与候选伪标签类别 c 相同；
2. teacher confidence >= 对应类别自适应 confidence threshold；
3. 邻居 feature 也落在候选类别 c 的 source Gaussian 区域内。

距离使用 L2-normalized feature 的 squared Euclidean distance：

```text
d_ij^2 = 2 - 2 cosine(z_i, z_j)
w_ij = exp(-d_ij^2 / (2 * h^2))
h^2 = bandwidth_multiplier^2 * global_var
```

## 密集 / 离散和最终分数

```text
density = mean_j(w_ij)
weighted_support = sum_j(w_ij * support_ij) / sum_j(w_ij)

dense = density >= density_threshold

density_factor =
    density                         if dense
    density * sparse_penalty        if sparse

score =
    G_c(z_i)
    * weighted_support
    * density_factor
```

因此：
- 邻居类别错误、置信度不足或不在 source Gaussian 区域，会降低 support；
- 邻域本身离候选较远会降低 density；
- sparse 邻域再乘 `sparse_penalty`；
- 最终 `score >= score_threshold` 才通过 reliability gate。

默认：
- `k=20`
- `score_threshold=0.5`
- `density_threshold=0.5`
- `sparse_penalty=0.5`

原有 OR rescue 语义保留：双弱视图一致但 confidence 不够的样本，只有通过上述 reliability score 才能被救回。

## 运行

```bash
git fetch origin
git checkout experiment/source-global-gaussian-knn-reliability-v7

mkdir -p new_logs
set -o pipefail

python -u train.py \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 \
  --pre_epochs 30 \
  --epochs 30 \
  --knn_gate \
  --knn_k 20 \
  --knn_distribution_mass 0.95 \
  --knn_variance_floor 1e-4 \
  --knn_sigma_momentum 0.70 \
  --knn_sigma_momentum_end 0.95 \
  --knn_sigma_momentum_ramp_refreshes 30 \
  --knn_bandwidth_multiplier 1.0 \
  --knn_density_threshold 0.5 \
  --knn_sparse_penalty 0.5 \
  --knn_score_threshold 0.5 \
  --knn_warmup_epochs 3 \
  --knn_refresh_interval 1 \
  2>&1 | tee "new_logs/cast_source_global_gaussian_knn_v7_$(date +%Y%m%d_%H%M%S).log"
```

如果已有 source-stage checkpoint，可以跳过 source pre-training：

```bash
python -u train.py \
  --checkpoint /path/to/source_checkpoint.pth \
  --pre_epochs 0 \
  --epochs 30 \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 \
  --knn_gate \
  --knn_sigma_momentum 0.70 \
  --knn_sigma_momentum_end 0.95 \
  --knn_sigma_momentum_ramp_refreshes 30 \
  --knn_sparse_penalty 0.5
```

## 单元测试

```bash
python -m unittest tests.test_global_gaussian_knn_reliability -v
python -m unittest discover -s tests -v
```

新增测试覆盖：
- source 真标签构建每类 center；
- 单一共享 global variance；
- global variance EMA momentum 随 refresh 增加；
- query 和 neighbor 都必须处于 source Gaussian 类区域；
- sparse neighborhood 的 score penalty。
