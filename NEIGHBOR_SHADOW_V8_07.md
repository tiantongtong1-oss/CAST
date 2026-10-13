# 邻域只读诊断
基于 v8-06 的 a900597。不启用 --history_neighbor，使用 --neighbor_diagnostics 时维持 v8-05 的原有半径救回（不延迟救回），邻域只记录不会拦截。默认不启用即时原型更新即为 A 更新设置。
CSV 保存在 models/rafdb_fer/<run_name>_neighbor_diagnostics.csv，每轮 flush。仅记录经过源域半径的低置信度救回候选。历史不足或无竞争类别的指标为 NaN，不把缺失当作零或无限间隔。local_margin 是 top-k 中已出现类别的平均余弦间隔，不是每类独立 top-k 检索。
新增 --neighbor_diagnostic_labels 可选择读取训练样本已有真实标签，仅写入 CSV，不参与筛选、损失、anchor、阈值或原型更新。这是带额外真实标签的机制诊断，不能声称无监督选参；若严格不使用目标训练标签，不启用该开关。当前未新增独立验证集邻域评估，不使用测试标签。
日志 Neighbor_Pass 在 shadow 模式指“假如启用审核会通过”的数量，不等于实际 Radius_Rescue_Num。

```bash
python -u train.py \
 --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
 --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
 --target_path /workspace/ttt/code/data/fer2013 \
 --pre_epochs 0 --epochs 10 --source_radius_quantile 0.90 \
 --neighbor_diagnostics --neighbor_diagnostic_labels \
 --run_name neighbor_shadow_a
python analyze_neighbor_diagnostics.py models/rafdb_fer/neighbor_shadow_a_neighbor_diagnostics.csv
```
移除 --neighbor_diagnostic_labels 可只观察信号，无法比较正确/错误标签。禁止同时启用 --history_neighbor 与 --neighbor_diagnostics。
分析统计按 sample-epoch 行汇总，不是独立样本；同样本多次出现和类别比例影响汇总。需做按 epoch/类别的分析与等接纳数量对照后再判断收益。
验证：NumPy 测试、CLI 分支解析和语法检查；无 PyTorch 环境，未做端到端训练。
