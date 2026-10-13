# 按类别邻域诊断 v8-08
基于 v8-07 c5099dd。--class_neighbor_diagnostics 自动启用 shadow 诊断，不能与 --history_neighbor 同用。历史排除不影响训练；CSV 保留历史指标供对照。最终接纳始终沿用 A 的 confidence OR radius rescue，默认仅 anchor 写入原型，不增加邻域训练门槛。

## 证据
每类分别检索 k 个余弦最近邻，记录均值 a_c；间隔是 a_pseudo - max(a_other)。源域真实标签库和目标 anchor 伪标签库分别计算。每类至少 k 个有效邻居（目标查询排除自身）才给完整间隔，否则 NaN。记录各类库数量和相似度，即使全局 top-k 都同类也能看到竞争类别。远距离竞争者也被检索，绝对相似度须结合间隔阅读。
源域邻居在原 source refresh 推理中收集，不额外增加图像增强或改变训练随机数。source_refresh_interval=1 每轮刷新；设置更大间隔会缓存源库。目标库取上一轮冻结 anchor 最新缓存特征，不要求四轮历史才能参与按类检索；存在跨 batch/epoch Teacher 版本陈旧性，不是源目标同一时刻编码。
目标真标签仅在 --neighbor_diagnostic_labels 打开后写 CSV，禁止进入筛选或损失。这是额外标签机制诊断，不是无监督选参证据；未增加目标验证集或测试集标签使用。

## 分析
analyze_class_neighbors.py 按 epoch 和预测类别报告源/目标 margin 有效率、分位数和正确/错误分布。等预算比较只使用四个分数均有限的共同候选池，按 25/50/75% 的相同数量选取 confidence 高、source_distance 低、source_margin 高、target_margin 高的样本。并列按 sample ID，报告共同池占比。仅覆盖原距离救回候选，不能推广为全部训练伪标签；小类别和共同池缺失须单独解释。跨轮行不是独立样本。

```bash
python -u train.py \
 --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
 --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
 --target_path /workspace/ttt/code/data/fer2013 \
 --backbone mobilenet_v2 --pre_epochs 0 --epochs 10 \
 --source_radius_quantile 0.90 --source_refresh_interval 1 \
 --class_neighbor_diagnostics --class_neighbor_k 10 \
 --neighbor_diagnostic_labels --run_name class_neighbor_shadow_a
python analyze_class_neighbors.py models/rafdb_fer/class_neighbor_shadow_a_neighbor_diagnostics.csv
```
验证：9 个 NumPy 单元测试（含暴力近邻对照、自身排除、缺失类、无效特征）、CLI 参数验证、分析脚本合成 CSV 检查、语法和 whitespace 检查。编辑环境无 PyTorch/FER 数据，不能保证端到端无问题或宣称效果提升。
