# Prototype KNN Reliability v4

> 本文保留 v4 历史设计。当前训练逻辑、参数、评分公式及运行方式见 [EXPERIMENT_SAMPLE_GAUSSIAN_KNN_V6.md](EXPERIMENT_SAMPLE_GAUSSIAN_KNN_V6.md)。

基于 `experiment/prototype-consistency-v3`，将能量拯救替换为原型置信域与目标域 k 近邻支持评分。保留 EMA Teacher、双弱视图一致性、类别自适应置信度阈值、原型一致性损失及 v3 的 OR 拯救规则。

## 评分定义

所有评分特征都先 L2 归一化。

1. 每次刷新从 `PrototypeBank.blended_prototypes()` 取一次原型快照 `mu[c]`，刷新间隔内固定评分中心；用于原型损失的原有目标原型仍按原流程更新。
2. 扫描全部源域训练样本，统计相对其各自类别中心的平方距离均值 `observed_var`。第一次直接初始化 `global_var`；以后每完成一次完整扫描更新 `global_var = beta * global_var + (1-beta) * observed_var`，`sigma = sqrt(global_var)`。
3. 用同一轮 EMA Teacher 的确定性视图建立整个目标训练集特征库。缓存特征、Teacher 预测类别、置信度和稳定 ID，不使用目标域真实标签。
4. 在全类别目标库中查找 kNN，并按 ID 排除自身。低置信样本仍参与检索，但不能提供类别支持；不在检索前筛成同类或高置信子集。

对于伪标签为 c 的候选样本 i：

```text
region(c, z) = ||z - mu[c]|| <= lambda * sigma
h = bandwidth_multiplier * sigma
w_ij = exp(-||z_i - z_j||^2 / (2 * h^2))
b_ij = (pred_j == c) AND (conf_j >= class_threshold[pred_j]) AND region(c, z_j)

density_i = mean_j(w_ij)
score_i = region(c, z_i) * mean_j(w_ij * b_ij)
```

分母是 k，不是权重之和，保留对稀疏邻域的惩罚。`score` 和 `density` 是 [0,1] 内的支持指标，不是校准后的正确概率或标准化概率密度。`sigma` 是径向 RMS 尺度，`lambda=2` 不表示 95% 覆盖率。

`density >= knn_density_threshold` 记为密集，否则记为稀疏；该阈值仅用于诊断，不额外参与筛选。密集与否相对于配置的带宽，不能解释为绝对统计结论。统计量缺失、原型无效或排除自身后不足 k 个有效近邻时不触发拯救，也不计入疏密统计。

## 如何接入训练

```text
m_conf = 两个弱视图类别一致，且两个置信度均超过原有类别阈值
m_knn  = 已成功评分 AND score >= knn_score_threshold

预热期间：m_final = m_conf
预热结束：m_rescue = m_agree AND m_knn AND NOT m_conf
          m_final = m_conf OR m_rescue
```

该实验替换的是拯救通道，不额外否决原有高置信样本。`con_idx` 保持 0/1，因为 `Networks.forward()` 使用 `idx == 1`；评分不直接作为 `idx`。获救样本继续参与分类、CAST affinity、原型损失和目标原型更新。所有统计和 kNN 评分均在 `no_grad` 下执行。

目标库每次刷新覆盖 target **train**，不读取 val/test 来建库。查询使用当前两个弱视图融合特征，库使用上次刷新的确定性特征，因此会存在增强差异和刷新间隔内的 Teacher 漂移。默认每 epoch 刷新一次，增大间隔会增加这种滞后。源域和目标域之间的偏移可能使原型范围过严；密集但整体预测错误的目标簇仍可能被高估，需要实验验证。

## 参数与运行

这些是实验起点，尚未通过 RAF-DB → FER2013 完整训练调优：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `knn_k` | 20 | 近邻数量 |
| `knn_interval_lambda` | 1.5 | 原型范围系数 |
| `knn_sigma_momentum` | 0.9 | 每次完整刷新更新一次的方差动量 |
| `knn_bandwidth_multiplier` | 1.0 | 核带宽相对 sigma 的倍数 |
| `knn_score_threshold` | 0.5 | 拯救评分门槛 |
| `knn_density_threshold` | 0.5 | 仅用于疏密诊断 |
| `knn_warmup_epochs` | 3 | 前 3 个目标 epoch 只统计，不拯救 |
| `knn_refresh_interval` | 1 | 以目标 epoch 为单位的刷新间隔 |
| `knn_query_chunk_size` | 128 | 查询分块大小，控制临时距离矩阵显存 |

从头训练（替换数据路径）：

```bash
python train.py \
  --source_path /path/to/raf-basic \
  --target_path /path/to/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 30 --epochs 30 \
  --knn_gate --knn_k 20 --knn_interval_lambda 1.5 \
  --knn_sigma_momentum 0.9 --knn_bandwidth_multiplier 1.0 \
  --knn_score_threshold 0.5 --knn_warmup_epochs 3
```

已有同骨干源模型时，使用 `--checkpoint /path/to/source_best.pth --pre_epochs 0` 跳过源域预训练。该方式只加载模型权重，为目标训练新建优化器，不是断点续训。需要完整恢复训练时仍须恢复 epoch、优化器、scheduler、EMA Teacher、PrototypeBank 和 reliability state，并重新建目标特征库。

用 `--no_knn_gate` 运行严格双视图置信度基线。v3 的 `--energy_*` 参数不再用于本分支；复现旧能量实验请使用原 v3 分支。

精确 kNN 每批计算分块距离矩阵 `[chunk_size, target_count]`，不建立全量 `[N,N]` 矩阵；总体计算量仍随目标训练集规模增长。不要为了提速直接删掉低置信样本或其他类别，这会改变邻域定义。

## 日志和保存

记录 `KNN_Source_Count`（日志标签为 `KNN Source_Count`）、`Target_Memory_Count`、`Global_Sigma`、`Sigma_Ready`、`KNN_Pass_Num`、`KNN_Rescue_Num`、`Final_Accept_Num`、`Mean_Reliability`、`Mean_Density`、`KNN_Checked_Num`、`Dense_Num`、`Sparse_Num`，并保留逐类伪标签分布。

新模型文件后缀为 `_prototype_knn_v4_source_best.pth` / `_prototype_knn_v4_target_best.pth`。目标检查点保存 `reliability_bank` 的原型快照、初始化状态、全局方差和源样本数，参数保存在 `args`。目标特征缓存不是持久状态，重新建库后才能启用拯救。

## 验证

```bash
python -m unittest discover -s tests -v
python train.py --help
```

测试覆盖固定支持比例下的疏密评分、跨类近邻、低置信近邻、自身 ID 排除、原型范围、初始化回退、方差 EMA、无梯度、分块等价性、OR 规则、稳定样本 ID、建库不使用目标真值和检查点保存。测试在 CPU 合成数据上执行；不代表完整训练精度验证。

实验比较建议：v3 能量拯救、v4 `--no_knn_gate`、v4 近邻拯救。检查相同接纳比例下的伪标签准确率、每类获救数量和验证性能；不要使用最终测试集调参。排序文件列表和改用局部数据集 RNG 会改变旧实验的随机轨迹，严格对照应在各实验使用相同的数据 ID/RNG 处理。
