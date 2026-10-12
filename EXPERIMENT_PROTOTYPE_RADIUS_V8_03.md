# v8-03：源域原型 + 最远 k 个距离均值

基于 `experiment/role-separated-sparse-knn-v8-02` 的提交
`d6224f5d87d09e119cd1c39dc51ea3ad49274458`。
新分支：`experiment/role-separated-sparse-knn-v8-03-prototype-radius`。

## 实现

将“k(比如20)和最远的距离取平均”解释为：取最远的 k 个样本距离求均值。
每次源域刷新，使用当前 EMA Teacher、原有确定性源域 loader 和源域真实标签：

1. 对特征做 L2 归一化，得到 z_i。
2. 每类原型 mu_c = mean(z_i | y_i=c)。mu_c 本身不再归一化，和原源域均值定义一致。
3. 计算所有该类源域样本的欧氏距离 d_i = ||z_i - mu_c||_2。
4. R_c = mean(topk_largest(d_i, min(source_radius_k, N_c)))。
5. 伪标签为 c 的目标样本：只有 ||normalize(z_t) - mu_c||_2 < R_c 才属于该类区域。

例如第 6 类，只比较第 6 类的原型和 R_6，不使用其他类别的阈值。
类别编号沿用数据集现有映射；代码索引从 0 开始，“第六类”对应索引 5。
`--source_radius_k 20` 独立于目标域邻居数 `--knn_k 20`。
少于 k 个源域样本时使用全部有效样本；类别缺失则本次刷新禁用该类别区域。
单样本或完全重合特征的 R_c=0，严格小于规则不接纳任何样本，不额外放宽阈值。
非有限特征、零向量和越界标签均不能通过。

`prototype_radius_reliability.py` 是新的运行时分布库，源域归属不再计算方差、
方差 EMA、卡方阈值或马氏距离。源域特征暂存在 CPU，确定最终均值后分批计算距离，
每类只保留当前最远 k 个距离；刷新完成清理临时源域特征。
阈值每次直接重估，不做 EMA。源域和目标域使用同一轮 Teacher 特征空间。
目标域标签、目标域特征均不参与原型区域阈值拟合。

原 `PrototypeBank` 的源域/目标域原型、原型 EMA 和一致性损失保留。
区域判断用当前源域均值，不使用目标域更新后的混合原型，避免伪标签改变自身接纳范围。

## kNN 接线与接纳规则

保留原分支的双视图一致性、高置信度 OR kNN 拯救、稳定角色晋升及即时原型更新开关。
新的区域判断替换原高斯判断，应用于 kNN 查询、邻居、sender 快照及 anchor 晋升。
**高置信度路线仍沿用原规则，可直接学习；区域检查并不是全部伪标签 CE 的统一额外门槛。**
属于区域仅表示满足几何条件；kNN 拯救还要满足支持数量、有效支持、纯度、margin 等原有条件。

kNN 权重也不再依赖方差：邻居所属类的带宽 h_c = bandwidth_multiplier * R_c；
权重 exp(-||z-z_j||² / (2*h_c²))。support 模式邻居距离上限为 radius_multiplier * R_c
（代码中比较平方值），默认 multiplier=2。极小尺度仅为权重除法提供 eps 保护；
源域归属 R_c 本身没有 floor。v7 评分消融仍可用，带宽也改为候选类别 R_c。
旧 `global_gaussian_knn_reliability.py`、`knn_reliability.py` 及对应历史测试保留用于历史实验，
新 `train.py` 和 `SparseReliableKNNBank` 不再调用它们。
日志输出 `Class_Radii`，checkpoint 保存 `class_means`、`class_radii` 和初始化状态。

## 运行命令

在已有 CAST 仓库根目录运行，替换数据与 checkpoint 路径：

```bash
git fetch origin
git switch --track origin/experiment/role-separated-sparse-knn-v8-03-prototype-radius
python -u train.py \
  --checkpoint /path/to/source_best.pth \
  --source_path /path/to/raf-basic \
  --target_path /path/to/fer2013 \
  --data1 rafdb --data2 fer \
  --backbone mobilenet_v2 --pre_epochs 0 --epochs 30 \
  --source_radius_k 20 --knn_k 20 \
  --immediate_knn_prototype_update \
  --run_name role_sparse_v8_03_prototype_radius
```

backbone 必须匹配 checkpoint。checkpoint 仅初始化模型权重，不是完整断点恢复。
若从头预训练，删除 `--checkpoint` 并设 `--pre_epochs 30`。
即时更新开关与 v8-02 文档示例一致；若对照实验没开该功能，这里也删去该开关。

```bash
python -m unittest discover -s tests -v
```

新增测试覆盖最远 k 个距离均值、分批一致性、少样本、零半径、严格边界、
类别缺失和刷新替换、无效输入，以及新 checkpoint 无方差字段。
更新 support 测试以使用原型半径，并保留合成数据训练与角色测试。
真实 RAF→FER 训练需要用户数据与 GPU；合成测试不代表性能改善。

验证结果：55 项 CPU 单元与合成训练测试全部通过，Python 语法检查通过。
