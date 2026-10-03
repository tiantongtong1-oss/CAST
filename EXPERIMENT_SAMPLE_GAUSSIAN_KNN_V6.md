# Sample Gaussian KNN reliability v6

基于 v5，可靠性模块完全移除混合类别原型及其归属判定。源/目标混合原型仍用于独立的原型一致性训练，不再作为候选样本是否能够获救的前提。

## 判定与评分

每次刷新根据 EMA Teacher 的归一化源域特征和源域真实标签，估计每类均值 mu[c] 和逐维方差 var[c]，构成对角高斯分布。均值与方差沿用 v5 的矩匹配 EMA、方差下限和缺失类别保护。目标域真实标签不参与拟合或评分。

```text
D_c(z) = sum_d((normalize(z)[d] - mu[c,d])^2 / var[c,d])
G_c(z) = valid_class(c) AND D_c(z) <= chi_square_quantile(D, distribution_mass)
b_ij = (pred_j == c) AND (conf_j >= threshold[c]) AND G_c(z_j)
score_i = G_c(z_i) * mean_j(w_ij * b_ij)
w_ij = exp(-||z_i-z_j||^2 / (2*h^2))
```

候选样本及邻居都使用候选伪标签 c 对应的源域分布检查。kNN 在全部目标训练集缓存中检索，按稳定 ID 排除自身，分母为 k。带宽仍为逐类方差迹的样本加权 RMS 乘 knn_bandwidth_multiplier。默认接纳质量 0.95、可靠性分数阈值 0.5。分数是支持指标，不是正确概率；卡方区域仅有模型假设下的名义覆盖意义。

保持 OR 拯救和预热逻辑：原有双视图高置信样本保留；一致但低置信的候选样本可通过该分数获救。源域统计不足、无效特征或有效近邻不足 k 时不能拯救。

## mu 输出

每次分布刷新自动打印每个有效类别完整的 mu 向量（不省略维度），以及 class、count、l2、abs_max、finite、bound_ok。无有效分布的类别打印 ready=False，不把旧均值当作当前有效统计输出。

由于特征先做 L2 归一化，均值及其凸组合 EMA 理论上满足 ||mu||_2 <= 1、max(abs(mu)) <= 1。检查允许 1e-5 浮点容差。超界或非有限时打印 Gaussian mu WARNING；诊断不会修改或截断均值。完整向量会增大日志，仅每次刷新打印，不在每个训练 batch 打印。默认源域预训练结束、进入目标域训练首次刷新后才出现 mu 日志。

## 运行

```bash
mkdir -p new_logs
set -o pipefail
python -u train.py \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 30 --epochs 30 \
  --knn_gate --knn_k 20 --knn_distribution_mass 0.95 \
  --knn_variance_floor 1e-4 --knn_sigma_momentum 0.9 \
  --knn_bandwidth_multiplier 1.0 --knn_score_threshold 0.5 \
  --knn_warmup_epochs 3 --knn_refresh_interval 1 \
  2>&1 | tee "new_logs/cast_sample_gaussian_knn_v6_$(date +%Y%m%d_%H%M%S).log"
```

检查点使用 `_sample_gaussian_knn_v6_source_best.pth` / `_sample_gaussian_knn_v6_target_best.pth` 后缀。reliability state 不再含 prototypes、prototype_initialized、prototype_in_distribution，旧 v5 reliability state 不能直接严格加载；可通过 --checkpoint 加载同骨干模型权重并重新拟合统计，仍非完整断点续训。

验证：20 项 CPU 单元/集成测试及 train.py --help 通过。包括无需原型即可评分、分布/邻域筛选、均值完整输出和超界报警。尚未执行完整数据集训练。
