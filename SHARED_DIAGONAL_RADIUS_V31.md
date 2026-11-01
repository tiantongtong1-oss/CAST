# 类别平衡共享对角马氏距离（v31）

基于 v30 提交 7ad3ba3c85ebc32cd8f0c7bc0991f1f088d6daa1。

## 统计与判断

当前 EMA Teacher 提取源域 L2 归一化特征，各类别保留自己的算术均值原型，
原型均值不再归一化。只用源域真实标签，不读取目标真实标签。

对每个特征维度 j：

- v_j = mean_over_present_classes(mean_over_class_samples((z_j - mu_cj)^2))。
- d(z,c) = sqrt(sum_j((normalize(z)_j - mu_cj)^2 / (v_j + lambda)))。
- 对源样本到真实类别原型的 d，按每类等总权重取90%经验分位，得到公共 T。
- 目标样本判断 d(z,pseudo_label) < T，边界严格拒绝。

七类共享一组方差与一个阈值，class_radii 的七个元素均保存 T。
方差是各类总体类内方差的等权平均，不包含类别均值之间的差异。
源域样本数量不会改变类别总权重。缺失类别不参与统计，几何判断拒绝；
全空刷新时均值、方差和阈值清零。零方差维度由 lambda 稳定。

参数：

```bash
--source_radius_quantile 0.90
--source_variance_regularizer 0.0001
--source_refresh_interval 1
```

lambda 是绝对加性方差正则，不是标准差，也不按类别分别设定。
旧 --source_radius_std_scale 已移除。本次不再使用均值加 k 倍标准差。
公共阈值没有跨轮平滑，不保证目标域覆盖率或每类各自90%覆盖率。
源域严格覆盖率受阈值同值、浮点舍入影响。

## 作用范围

马氏距离替代 radius 分支及 anchor 几何资格判断中的欧氏距离。
KNN仍为单位特征空间的欧氏11邻居、至少6票；competition仍使用原来的
欧氏原型差值0.02、概率差0.1、票数差3。高置信直接接纳路径保持不变。
因此这不是把所有距离运算都替换为马氏距离。

日志新增共享方差最小/最大值和 Variance_Regularizer，模式为
class_balanced_shared_diagonal_mahalanobis。checkpoint reliability_bank 新增
shared_variance 与 variance_regularizer buffers。不能将旧 reliability_bank
直接 strict-load 到新银行；训练入口 --checkpoint 只加载 model 权重，
会重新拟合所有几何统计，因此可以复用旧的 source_best.pth。

## 验证

```bash
python -m unittest discover -s tests -p test_shared_diagonal_radius.py -v
python -m unittest discover -s tests -p 'test_balanced*radius.py' -v
python -m unittest discover -s tests -p test_global_mean_std_ema.py -v
python -m unittest discover -s tests -p test_neighbor_vote.py -v
python -m unittest discover -s tests -p test_competition_gate.py -v
python -m py_compile shared_diagonal_radius.py source_prototype_radius.py train.py
```

当前开发环境没有 PyTorch，银行刷新/查询/保存集成测试跳过；已验证实际
银行所调用的 NumPy 拟合核心，未执行GPU训练。历史 test_agreement_radius_ds.py
在v29/v30已有的失败不属于本次变更。


## OR 晋升更新

晋升资格改为 `agreement & (confidence | radius | neighbor)`。
neighbor 是完整11个源域邻居中至少6票支持当前伪标签，不要求竞争检查通过。
高置信样本已经满足OR，无需额外查询邻居；其余双视图一致样本每轮查询一次，
供晋升和救援复用，包括救援尚未开启的前3轮。
连续3个epoch有资格且伪标签不变才成为anchor；资格失效清零，标签变化重新累计。
各轮可以通过不同证据路径维持连续计数。

救援仍受warmup和competition约束，实际分类/原型一致性监督仍为
`confidence | (rescue & anchor)`。不通过最终接纳的样本即使晋升，也不会
因此直接进入这些监督损失；但anchor本身可参与目标原型更新及anchor相关损失。
这是放宽晋升资格的预期效果，并非将竞争检查改成OR。

日志新增 Promotion_Eligible_Num，并打印完整晋升公式。
验证：`python -m unittest discover -s tests -p test_promotion_or.py -v`。


## 限制目标原型写入

目标原型更新条件收紧为 `prototype_update_mask = anchor_mask & final_mask`。
晋升后但本轮未最终接纳的样本不再写入目标原型。OR晋升资格、监督掩码及其他
anchor相关操作不变。比较时使用同一源域checkpoint，设置promotion_epochs=2、
threshold_base=threshold_min=0.875，其余参数保持一致。
新增测试 test_accepted_anchor_updates.py 验证实际训练掩码及更新调用。
