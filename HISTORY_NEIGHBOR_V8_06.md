# 历史分组排除邻域救回（实验功能）

基于 experiment/source-radius-quantile-v8-05 的 03378311。
开启 --history_neighbor 后，仅改变 radius rescue：双视图一致、未通过高置信度且通过源域半径的候选，还必须通过历史邻域审核。高置信度 OR 接纳、Student 全 batch 前向、损失掩码、MMD anchor 晋升及原型更新开关沿用基线。

## 算法
- 按稳定样本 ID 保存每轮 Teacher 双弱视图融合特征、预测类别、最终接纳、anchor 状态、双视图最低置信度。不读取目标真实标签。
- 每轮开始冻结上一轮记忆，只有上一轮 anchor 且最近 T 轮均被观察的样本可作邻居。不是同一个 Teacher 时间点全量重新编码的特征，存在跨 batch 特征陈旧性。
- 余弦相似度不低于阈值的 top-k 邻居，排除自身。
- 分组签名：T 轮标签、接纳和 anchor 状态，以及置信度变化三档（小于 -0.05、[-0.05,0.05]、大于 0.05）。按完全相同签名分组。
- 按签名 Hamming 距离排除与查询历史最近的一组；并列采用邻居相似度排序后首次出现的组。
- 剩余每组先计算支持当前伪标签的比例，再对组等权平均。单个组增加重复成员不会增加其组权重；新增样本仍可能改变 top-k 构成。
- 默认剩余至少 3 个邻居、2 组，支持度 >=0.8；否则不救回。不回退到无邻域的距离救回。
- 历史不足时仅高置信度接纳。因此与基线比较时也需注意额外 warmup 的影响。
- 历史状态随 checkpoint 保存；原训练入口仍仅从 checkpoint 加载模型权重，未新增完整恢复训练功能。

## 局限
历史签名只是相关性代理，不能识别错误因果来源或保证独立证据。正确同步变化也可能被拒绝；稳定同类样本的历史可能全部相同而无剩余组。须观察 Evidence_Ready、Neighbor_Pass 和每类接纳分布，若长期接近零则本机制退化为高置信度自训练。阈值为实验起点，不是已优化参数。CPU 查询可能增加训练时间。

## 运行
```bash
mkdir -p test_logs
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --pre_epochs 0 --epochs 30 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --history_neighbor --neighbor_history_window 4 --neighbor_k 20 \
  --neighbor_support 0.8 --neighbor_min_remaining 3 \
  --neighbor_min_groups 2 --neighbor_min_similarity 0.5 \
  --run_name history_neighbor_q90_anchor \
  2>&1 | tee test_logs/history_neighbor_q90_anchor.log
```
此命令不启用 immediate_prototype_update，使用仅 anchor 更新。若对比 B，应在双方一致添加该参数。不加 --history_neighbor 即为原基线行为。

## 验证
```bash
python -m unittest discover -s tests -p test_history_neighbor.py
python -m py_compile train.py history_neighbor.py
```
已运行独立 NumPy 单元测试与语法检查；编辑环境无 PyTorch/数据，未运行端到端训练，不宣称准确率提升或首创。
