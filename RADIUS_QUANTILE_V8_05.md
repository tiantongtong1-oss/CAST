# v8-05：源域分位数半径

基于 experiment/dual-view-confidence-radius-rescue-v8-04 的 d3392ccd。
每类源域有效样本先做 L2 归一化，再求类均值（不再次归一化）。
收集整类所有样本到均值的欧氏距离，使用 torch.quantile(..., interpolation="linear")。
默认 --source_radius_quantile 0.90，合法范围 0 < q <= 1。
旧 --source_radius_k 参数已移除。历史 v8-04 文档中的最远 k 规则不适用于本分支。

目标仍按伪标签对应类别比较 distance < radius；并列距离和零半径时，实际覆盖率不保证恰好为 q。
源域真实标签决定统计分组，不要求模型预测正确；不使用目标真实标签。
每个 epoch 全量刷新，原有半径救援 warmup 和高置信度 OR 接纳逻辑保留。
半径还用于锚点晋升的 region_ok，因此更改半径也可能改变锚点数量。
本分支沿用远端 v8-04：--immediate_prototype_update 允许所有已接纳样本更新目标原型，
不是服务器本地可能已修改的“锚点 ∪ 救援样本”版本。没有新增 KNN。

## 运行
无需重新预训练源域；从原 source_best 权重开始目标训练。
在本分支目录中：
```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --pre_epochs 0 --epochs 30 \
  --source_radius_quantile 0.90 \
  --source_refresh_interval 1 \
  --immediate_prototype_update \
  --run_name confidence_radius_q90 \
  2>&1 | tee test_logs/confidence_radius_q90.log
```
其他数据路径沿用 train.py 默认值；如服务器数据位置不同，显式传 --source_path 和 --target_path。
q=0.95 时同时更换 run_name 和日志名，避免覆盖 q90 结果。

## 验证
已通过 Python 语法检查及独立提取 parse_args 的默认值、覆盖值、非法值检查。
编辑环境未安装 PyTorch，未运行张量计算测试或真实训练，不宣称训练效果提升。
