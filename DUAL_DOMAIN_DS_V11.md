# v11：距离救回后的双域 DS 邻域审核

基于 `experiment/source-radius-quantile-v8-05` 的 `03378311db45515544632daa4c23087089963f9d`。
不包含 v9/v10 的实验机制。启用参数为 `--ds_neighbor`；不加时保留 v8-05 的接纳行为。

## 接纳顺序与角色

双弱视图预测一致且未达到置信阈值 → 本类源域 0.9 分位半径内 → 双域邻域证据审核 → 距离救回。
`final_mask = confidence_mask | (rescue_candidates & region_ok & ds_ok)`，仍有原 warmup。
高置信路径不经过 DS；不能描述为“全部最终样本均经过 DS”。
原 anchor 连续晋升、MMD 的 anchor_mask、原型 A/B 参数保留。
推荐命令不传 `--immediate_prototype_update`，即仅 anchor 更新目标原型（A）。
不可靠样本的损失 mask 和前向行为仍沿用基线，本改动不保证被拒样本完全不影响 BatchNorm。

## 参考库

每个 target epoch 开始，以当前 EMA Teacher 的 eval 特征重新提取全部源域训练样本，
以及上一轮已经连续晋升的目标 anchor，整轮冻结。目标库使用确定性 test transform，
用 `sender_scores` 检查当前预测仍与历史标签一致，质量取当前置信度与历史质量的较小者。
目标真实标签从不用于审核。数据路径列表必须与训练集完全一致；源域与目标域 ID 不共享命名空间。
邻居库每 ID 一份；目标查询排除自身。两个弱视图不当作两条独立证据。
每轮重建库增加一次源域和一个 anchor 子集的前向开销；不是精确断点续训。

## 计算

查询为原有的双弱视图归一化平均特征；两域特征均逐样本 L2 归一化。
在所有类别的参考样本中取距离不超过 `ds_max_distance` 的最近 `ds_k` 个，不预先按查询伪标签筛选。
这是查询中心的局部最大距离，与源域原型的类半径不是同一量。
当前实现为固定的查询邻域最大距离，不宣称自适应邻域半径。
源标签质量 q=1，目标 q 使用上述稳定 anchor 质量；r=min(q exp(-d²/(2h²)), strength_cap)。
每个邻居的单类质量为 r，Omega 质量为 1-r。
域内执行普通 Dempster 融合，用未归一化合取计算并保留总冲突 K，避免只记录最后一步冲突。
不足 min_neighbors、总冲突过高、无效特征时，域不可用并返回空证据 Omega=1。
单个邻居 strength_cap 防数值退化，并不解决相关证据重复累计；无谨慎融合或冗余分组。

两域之间为线性池：m=alpha*m_source+(1-alpha)*m_target，**不是第二次 DS 组合**。
不因参考库大小或某域缺失自动重新分配权重。
跨域冲突=sum_{a!=b} source_mass[a]*target_mass[b]，只作矛盾指标，不表示错误概率。
严格要求两域可用，然后同时要求本类支持、与最强竞争类间隔、不确定质量、跨域冲突达到阈值。
域内与跨域冲突目前共用 ds_conflict_threshold。未通过表示暂不接纳，不是证明标签错误。

## 参数（未经目标实验校准的起始值）

- ds_k=20；ds_min_neighbors=3；ds_max_distance=0.5；ds_bandwidth=0.25。
- ds_source_weight=0.5；ds_strength_cap=0.95。
- ds_support_threshold=0.6；ds_margin_threshold=0.1。
- ds_uncertainty_threshold=0.3；ds_conflict_threshold=0.5。

`ds_k` 是近邻数量上限，与旧 v8-04 的最远 k 半径无关；本分支仍是 0.9 分位原型半径。
前几轮目标 anchor 不足，DS 会阻止救回，但高置信样本仍可训练及积累 anchor。
这些参数只是起点，不保证收益；不要用目标测试标签调参。

## 运行

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --pre_epochs 0 --epochs 30 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --ds_neighbor --ds_k 20 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 \
  --ds_source_weight 0.5 --ds_support_threshold 0.6 \
  --ds_margin_threshold 0.1 --ds_uncertainty_threshold 0.3 \
  --ds_conflict_threshold 0.5 \
  --run_name dual_domain_ds_q90_a \
  2>&1 | tee test_logs/dual_domain_ds_q90_a.log
```

数据路径沿用 train.py 默认值，可显式覆盖 --source_path/--target_path。
checkpoint 仅初始化权重，不恢复 optimizer、EMA、anchor 历史。基线对照用相同命令去掉 --ds_neighbor 并换 run_name。
使用新的 run_name，重复运行同名会覆盖对应诊断与 checkpoint。

## 输出与检查

- 每轮 DS_Source_Refs/DS_Target_Anchor_Refs。
- DS_Audited 是通过半径后的救回候选数；DS_Passed 是审核后救回数。
- 两域可用数、综合支持/不确定性/跨域冲突、域内冲突均值。
- `models/rafdb_fer/dual_domain_ds_q90_a_ds_diagnostics.csv`，仅记录被审核候选，逐样本保存两域支持、数量、可用性、不确定性、冲突、最终间隔与是否接受。没有目标真实标签。

```bash
python -m unittest discover -s tests -p test_dual_domain_ds.py -v
python -m py_compile train.py dual_domain_ds.py
```

新增数值、参数及接入检查在无 PyTorch 环境可运行。编辑环境没有 PyTorch、CUDA 和 RAF/FER 数据；
未执行真实训练或 Torch 数据加载联调，不宣称准确率提升。
