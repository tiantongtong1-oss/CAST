# 类别均值动态置信度 + 源域半径 / DS 补救

直接更新原分支 `experiment/agreement-radius-ds-no-conf-no-js-v18`。
历史提交 `87fbaa64b8e5308590a34f2872ccc39138b26e16` 是完全去掉置信度通道的版本。
当前版本恢复动态置信度直接接受通道，不使用 JS。

## 阈值

每个目标域 epoch 开始，当前 EMA Teacher 用两个独立弱增强视图扫描整个目标域训练集。
对双视图预测类别一致的样本，令 s = min(conf1, conf2)，按预测类别累计 s 的和及样本数。
阈值 tau[c] = sum[c] / count[c]，不是 batch 均值的平均，也不是在每个训练 batch 重算。
本轮阈值固定，下一轮从零统计。真实标签不参与统计、筛选或损失。
没有一致样本的类别阈值设为 +inf，暂时关闭该类直接接受通道。

不再需要 threshold_base/beta/margin/min/max 或 phi，不增加差值容差和融合系数。
直接接受条件：pred1 == pred2 且 min(conf1, conf2) >= tau[pred1]。

## 补救及训练

最终接受 = 动态置信度接受 OR (类别一致 AND 非置信度接受 AND 源域半径通过 AND DS通过)。
恢复原高置信度方案的前三轮补救 warmup：epoch 0–2 仅动态置信度直接接受；epoch 3 起启用补救。
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
  --run_name classmean_minconf_radius_ds_nojs_v18 \
  2>&1 | tee test_logs/classmean_minconf_radius_ds_nojs_v18.log
```

## 日志及诊断

- 每轮输出 Confidence_Thresholds、Agreement_Class_Counts、Threshold_Mode: class_mean_min_conf。
- Confidence_Accept_Num 为直接接纳数，Radius_Rescue_Num 为 DS 补救数，两者之和等于 Final_Accept_Num。
- Radius_Pass_Num 为未通过置信度但类别一致且半径内的候选数。warmup 期间可有候选，但不做 DS 补救。
- 分别打印直接接纳、补救、最终的类别分布。
- DS CSV 只包含进入 DS 的补救候选，不包含高置信度直接接受样本；无 JS 字段。
- inf 阈值表示该轮统计时此类别没有一致预测，不是计算错误。

## 检查

```bash
python -m unittest discover -s tests -p '*ds*.py' -v
python -m py_compile train.py adaptive_confidence.py tests/test_agreement_radius_ds.py
```

36 项针对性测试通过，包括不等长 batch 的统计、类别不一致排除、空类别、每轮统计重置、
直接接受绕过半径、低置信度 DS 补救、无候选、warmup、边界及 CSV 格式。
使用真实训练筛选片段、原有双视图选择函数及 DS 融合，NumPy adapter 代替 Tensor。
环境缺少 PyTorch 和 RAF/FER 数据，未运行端到端训练，不能据此宣称精度改善。
