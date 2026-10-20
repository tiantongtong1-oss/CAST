# 原动态阈值 + 双视图最小置信度 + 半径 / DS 补救

分支：`experiment/agreement-radius-ds-no-conf-no-js-v18`。
本次在 e4224b9 基础上恢复原先的五参数阈值，停止使用“双视图最低置信度类别均值”方案。

## 当前阈值公式

每个 epoch 开始，Teacher 用一个弱增强视图扫描整个目标域训练集。
按预测类别计算最大概率均值 mu[c]，再对有预测样本的各类别均值取平均 global_mean。

```text
center = min(max(threshold_base, global_mean + threshold_margin), threshold_max)
tau[c] = clip(center + threshold_beta * (mu[c] - global_mean), threshold_min, threshold_max)
```

无预测样本的类别设为 threshold_max。默认参数为 base=.85、beta=.5、margin=.02、min=.80、max=.95。
本轮训练 batch 共用阈值；不使用目标域真实标签、验证集或测试集计算阈值。

训练时仍使用两个弱视图，筛选显式写成：

```python
min_confidence = torch.minimum(conf1, conf2)
reliable = agreement & (min_confidence >= sample_threshold)
```

这与两个置信度分别 >= 同一个阈值数学等价；恢复旧阈值公式才是相对 e4224b9 的算法变化。
保留 >= 的包含边界行为，不改成严格 >。adaptive_confidence.py 为旧实验工具，不进入当前训练。

## 补救及训练

最终接受 = 自适应置信度接受 OR (类别一致 AND 非置信度接受 AND 源域半径通过 AND DS通过)。
恢复原高置信度方案的前三轮补救 warmup：epoch 0–2 仅自适应置信度直接接受；epoch 3 起启用补救。
可用 --radius_rescue_warmup_epochs 调整，默认 3。此调度独立于原型损失 warmup。
DS 始终作为补救审核启用，--ds_neighbor 是兼容参数。证据仅来自有真实标签的源域。
源域半径默认每轮刷新，DS 证据每轮刷新。高置信度样本不必通过半径或 DS 即可参与分类监督。
anchor 晋升仍要求半径内且连续满足条件；--immediate_prototype_update 允许全部最终接纳样本即时更新原型。
Student 和 EMA Teacher 每 batch 更新。无目标样本通过时，源域监督仍继续。

## 更新与运行

```bash
git switch experiment/agreement-radius-ds-no-conf-no-js-v18
git pull --ff-only origin experiment/agreement-radius-ds-no-conf-no-js-v18
```

使用同一个源域 checkpoint，启动新实验，不是断点恢复。路径按实际环境修改。
使用新的 run_name，避免覆盖旧版结果。

```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --pre_epochs 0 --epochs 30 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --radius_rescue_warmup_epochs 3 \
  --immediate_prototype_update \
  --ds_neighbor --ds_k 11 --ds_min_neighbors 3 \
  --ds_max_distance 0.5 --ds_bandwidth 0.25 --ds_strength_cap 0.95 \
  --ds_support_threshold 0.6 --ds_margin_threshold 0.1 \
  --ds_uncertainty_threshold 0.3 --ds_conflict_threshold 0.5 \
  --run_name adaptive_minconf_radius_ds_nojs_v18 \
  2>&1 | tee test_logs/adaptive_minconf_radius_ds_nojs_v18.log
```

## 日志及诊断

每轮日志输出 Confidence_Thresholds、Class_Mean_Confidence、Global_Class_Mean、Threshold_Center，
并显示 Threshold_Mode: original_adaptive_min_conf。
Confidence_Accept_Num 是直接接纳数，Radius_Rescue_Num 是补救数，两者相加为 Final_Accept_Num。
DS CSV 仅记录补救候选，不包含直接接纳样本；没有 JS 字段。

## 检查

```bash
python -m unittest discover -s tests -p '*ds*.py' -v
python -m py_compile train.py ema_utils.py
```

37 项针对性测试通过，包含显式最小置信度与双比较等价性、边界、补救及 warmup。
阈值扫描函数恢复自 v17。环境没有 PyTorch 和 RAF/FER 数据，未运行完整训练。
