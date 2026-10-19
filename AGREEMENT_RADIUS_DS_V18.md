# 双视图类别一致 + 源域半径 + Source-only DS（v18）

基于 `experiment/source-ds-js-late-v17`，提交 `13a46860368cb4efbfd1fb1bbedfd04f2d8cf2a1`。

## 筛选规则

`accepted = view_agreement AND source_radius_pass AND source_DS_pass`

- 删除高置信度直接接受通道、阈值统计扫描及 threshold/phi 参数。
- 两个弱增强视图预测类别一致后，用归一化融合 Teacher 特征检查该类源域半径，再用源域邻居 DS 审核。
- 高置信度也不能绕过半径/DS；低置信度可以在三个条件都通过后参与学习。
- DS 始终开启；`--ds_neighbor` 仅作为兼容命令保留，省略也开启。
- 删除训练中的 JS 计算、JS 启动调度、JS 参数和 CSV 字段。历史诊断工具及其独立 JS 数值函数保留，不进入训练筛选。
- 从目标阶段 epoch 0（第一轮）启用半径和 DS，删除原来的 rescue warmup。原型损失的 warmup/ramp 仍保留，与伪标签筛选不是同一机制。
- DS 证据库每轮开始刷新；源域半径按 source_refresh_interval 刷新，默认每轮一次。
- Student 每 batch 更新，Teacher 随后做 EMA；源域监督、原型损失、anchor 连续轮数晋升规则不变。
- conf1/conf2 仍用于记录 anchor quality，但不设高置信度门槛，不控制最终接受条件。
- 没有通过样本时，目标分类/原型损失贡献为零，源域监督仍继续。

## 切换分支

在本地 CAST 仓库中执行：

```bash
git fetch origin
git switch --track origin/experiment/agreement-radius-ds-no-conf-no-js-v18
```

## 运行

以下沿用 v17 文档的源域预训练 checkpoint 和 DS 设置（k=11）。确认 checkpoint 和数据目录在本机存在；如不同，修改路径。此命令从源域模型开始新的 30 轮目标域训练，不是恢复 v17 的 optimizer/epoch。

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --pre_epochs 0 --epochs 30 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --immediate_prototype_update \
  --ds_neighbor --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 --ds_strength_cap 0.95 \
  --ds_support_threshold 0.6 --ds_margin_threshold 0.1 \
  --ds_uncertainty_threshold 0.3 --ds_conflict_threshold 0.5 \
  --run_name agreement_radius_ds_no_conf_no_js_v18 \
  2>&1 | tee test_logs/agreement_radius_ds_no_conf_no_js_v18.log
```

## 检查

```bash
python -m unittest discover -s tests -p '*ds*.py' -v
python -m unittest discover -s tests -p 'test_rescue_js*.py' -v
python -m py_compile train.py dual_domain_ds.py tests/test_agreement_radius_ds.py
```

32 个 DS/诊断/新筛选测试和 5 个历史 JS 工具测试通过。新测试执行 train.py 实际筛选代码片段，使用 NumPy tensor adapter 和真实 DS 融合，覆盖低置信度通过、高置信度被半径/DS拒绝、类别不一致、无候选样本和 CSV 行格式。

当前检查环境没有 PyTorch、CUDA 和 RAF/FER 数据，未执行端到端训练或实测精度。仓库其他旧版本集成测试针对历史 KNN 接口，不代表 v18 的训练验证；以上命令是本次针对性检查。

观察 `Agreement_Num → Radius_Pass_Num → Final_Accept_Num` 和 `DS_Audited/DS_Passed`，其中 Radius_Pass_Num 只统计类别一致且通过半径的样本。CSV 每行对应进入 DS 的样本，`accepted = ds_pass`，不再有 JS 列。
