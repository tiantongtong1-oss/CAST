# Prototype Gaussian KNN Reliability v5

> 本文为历史 v5 设计。当前分支采用 [v6 逐样本分布判定](EXPERIMENT_SAMPLE_GAUSSIAN_KNN_V6.md)，已移除混合原型归属门控并增加完整 mu 日志。

基于 `experiment/prototype-knn-reliability-v4`。将统一球形范围替换为逐类对角高斯分布，分别判断伪标签对应的混合类别原型、候选样本和 k 近邻是否受到该类别分布支持，再计算可靠性分数。

## 类别分布

每次刷新使用同一 EMA Teacher 扫描全部源域训练样本。对 L2 归一化特征 `z`，按源域真实标签拟合每个类别：

```text
mu[c] = mean(z_i | y_i=c)
var[c,d] = mean((z_i[d] - mu[c,d])^2 | y_i=c)
p(z | c) = Normal(mu[c], diag(var[c]))
```

均值不再次归一化；高斯分布中心来自当前源域统计，与 `PrototypeBank.blended_prototypes()` 返回的源/目标混合原型是两个不同对象。后者记为 `p_c`，也必须接受分布检查，不能用它自身作为均值来检查自身。

统计使用 float64 累积，方差采用 Gaussian MLE（分母为 N），每类至少需要两个有效源样本。方差下限为 `knn_variance_floor`，避免零方差导致除零。刷新间使用 `knn_sigma_momentum=beta` 做矩匹配 EMA：

```text
mu_new = beta * mu_old + (1-beta) * mu_observed
var_new = beta * var_old + (1-beta) * var_observed
          + beta * (1-beta) * (mu_old - mu_observed)^2
```

最后施加方差下限。缺失或样本不足的类别在本轮禁用拯救，重新有足够样本时重新初始化该类统计。

## 统一归属判定

```text
D_c(x) = sum_d((normalize(x)[d] - mu[c,d])^2 / var[c,d])
q = ChiSquare(feature_dim).inverse_cdf(knn_distribution_mass)
G_c(x) = valid_class(c) AND D_c(x) <= q
```

这是由高斯密度等高面定义的椭球区域。默认 `knn_distribution_mass=0.95`，表示假设模型下的名义概率质量，并非实际目标域的 95% 覆盖率或伪标签正确率。L2 归一化特征位于单位球面，维度之间并不独立，因此对角高斯只是近似模型；需要用验证集评估迁移效果和阈值，不能把“在区域内”解释成已经证明属于该类。

原型 `p_c` 使用每次刷新时的混合原型快照；源域统计、原型归属和目标特征库在刷新间保持固定。

## kNN 与可靠性分数

保留全类别目标训练集检索和按稳定 ID 排除自身的逻辑，低置信样本也参与检索。源类别分布的拟合不使用目标域真实标签。

```text
sigma^2 = sum_c(N_c * sum_d(var[c,d])) / sum_c(N_c)  # 仅有效类别
h = knn_bandwidth_multiplier * sigma
w_ij = exp(-||z_i-z_j||^2 / (2*h^2))
b_ij = (pred_j == c) AND (conf_j >= threshold[pred_j]) AND G_c(z_j)
score_i = G_c(p_c) * G_c(z_i) * mean_j(w_ij * b_ij)
pass_i = enough_k_valid_neighbors AND score_i >= knn_score_threshold
```

邻居使用候选伪标签的类别 `c` 做分布判定，同时要求邻居 Teacher 预测也是 `c`。同类但分布外的近邻不能提供支持；原型或候选样本分布外时分数为零。分母为 k，保留对稀疏邻域的惩罚。`score` 在 [0,1] 内，是可靠性支持分数，并非校准的后验概率。`density=mean(w_ij)` 仍仅用于诊断。

训练继续采用 OR 拯救：预热后，双视图一致、未通过原有置信度筛选且通过上述分数的样本可以获救；原有高置信样本不被此分数额外否决。获救样本参与原有分类、CAST affinity、原型损失及目标原型更新。

## 运行与迁移

```bash
python train.py \
  --source_path /path/to/raf-basic --target_path /path/to/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 30 --epochs 30 \
  --knn_gate --knn_k 20 --knn_distribution_mass 0.95 \
  --knn_variance_floor 1e-4 --knn_sigma_momentum 0.9 \
  --knn_score_threshold 0.5 --knn_warmup_epochs 3
```

`--knn_interval_lambda` 已移除，请改用 `--knn_distribution_mass`；二者没有直接数值换算关系。质量参数越大，接纳范围越宽。保留 `--knn_sigma_momentum` 参数名，但它现在控制逐类均值和方差的矩匹配 EMA。

检查点后缀改为 `_prototype_gaussian_knn_v5_source_best.pth` / `_prototype_gaussian_knn_v5_target_best.pth`，保存逐类均值、方差、初始化状态、源样本数、卡方阈值和原型归属掩码。目标特征库不保存，恢复后需刷新建库。v4 reliability state 缺少这些统计，不能直接加载为 v5 分布；可通过原有 `--checkpoint ... --pre_epochs 0` 加载模型权重后重新建立统计，该选项仍不是完整断点续训。

日志新增逐类源样本数和 `prototype_membership`。`global_var` / `Global_Sigma` 保留为核带宽尺度，现在来自逐类方差迹的样本加权平均，不参与定义球形接纳范围。

## 验证

```bash
python -m unittest discover -s tests -v
python train.py --help
```

测试覆盖逐类均值/方差、含均值漂移的 EMA、已知卡方分位数、各向异性判定、原型漂移拒绝、样本与邻居归属、缺失类别和零方差、近邻自身排除、跨类/低置信近邻、分块与无梯度、训练刷新及检查点。CPU 合成测试不代表完整训练精度；尚未运行 RAF-DB → FER2013 完整训练。
