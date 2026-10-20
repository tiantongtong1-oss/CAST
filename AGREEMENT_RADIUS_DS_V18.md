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

## 平均邻居支持度补救（本次更新）

保留原双视图阈值公式、类别一致性、源域半径、三轮补救 warmup、EMA 及原型更新流程；继续无 JS。
高置信度直接接纳路径完全保留。

源域证据有真实标签。使用 Teacher 特征的余弦相似度选最近 K 个源域邻居，
每个邻居的强度 s=max(0, cosine)，其真实标签获得质量 s，未知集合获得 1-s。
外层等权平均，候选标签 A 的分数 S=mean(s * I(source_label=A))。
不再使用 Dempster 连乘融合；相同弱证据重复出现不会仅因数量增加而趋近 1。
本实现是平均证据支持审核，不是相关性修正 DS。

每轮刷新源域证据和所需半径后，额外扫描目标训练 loader 的双弱视图。
仅收集“类别一致、未被双视图置信度直接接纳、通过源域半径、邻居数量足够”的候选支持度。
对这些分数做精确 Otsu 分割，阈值取高分组最小值，训练 batch 使用 S>=阈值。
不读取目标真实标签。额外扫描会增加每轮计算量，随机增强及 Teacher batch 更新意味着校准分布与训练分数不完全一致。
无候选或所有分数相同时阈值为 inf，本轮关闭补救，直接置信度路径继续。
Otsu 仅划分相对高分组，不保证伪标签正确率。

邻居审核只使用 --ds_k 和 --ds_min_neighbors；保留旧 CLI 参数兼容性，但旧距离截断、
带宽、强度上限、support/margin/uncertainty/conflict 四阈值不再参与新审核。
旧参数仍经过兼容合法性检查，启动日志会明确说明它们不参与审核。
DSConfig 及 dual_domain_ds.py 中旧算法保留供旧实验和诊断工具调用。
最终接受 = 原双视图置信度接受 OR 平均支持度补救接受。

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
  --run_name adaptive_minconf_radius_mean_support_otsu_nojs_v18 \
  2>&1 | tee test_logs/adaptive_minconf_radius_mean_support_otsu_nojs_v18.log
```

## 日志及诊断

新增 Neighbor_Support_Otsu 和 Calibration_Count。
保留旧 CSV 列以兼容工具，但 source_support/support 现在是平均单类别质量，
unknown 是平均未知质量，margin 仅诊断，conflict 固定为 0（未计算 DS 冲突），均不单独设阈值。
历史 diagnose_ds.py 仍是旧 DS 诊断，不代表当前训练选择规则。

## 检查

```bash
python -m unittest discover -s tests -p '*ds*.py' -v
python -m unittest discover -s tests -p 'test_neighbor_support.py' -v
python -m py_compile train.py neighbor_support.py
```

环境没有 PyTorch 和 RAF/FER 数据，未运行完整训练。
