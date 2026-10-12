# v8-04：双视图高置信度 + 源域原型距离挽回

新分支：`experiment/dual-view-confidence-radius-rescue-v8-04`。
基于 v8-03 direct 分支提交 `b8332b7b21fc93e13db9bda688c4cb7ce613b9e5`。

## 接纳规则

对两个弱增强视图，Teacher 分别输出预测类别和最大 softmax 概率。
记 A 为两视图类别一致，H 为两视图置信度均 >= 预测类别阈值 tau_c，
G 为融合特征到预测类别源域原型的欧氏距离严格小于该类半径 R_c：

```python
confidence_mask = A & H
rescue_candidates = A & ~confidence_mask
rescue_mask = rescue_candidates & G   # 仅在挽回开启后
final_mask = confidence_mask | rescue_mask
```

| 双视图 | 两视图置信度 | 距离条件 | 结果（挽回开启后） |
|---|---|---|---|
| 不一致 | 任意 | 任意 | 拒绝 |
| 一致 | 均达到阈值 | 任意 | 直接接纳 |
| 一致 | 至少一个未达到阈值 | d < R_c | 挽回接纳 |
| 一致 | 至少一个未达到阈值 | d >= R_c | 拒绝 |

高置信度路线不受距离筛选否决。挽回路线不添加另一条置信度下限。
`--radius_rescue_warmup_epochs 3` 默认前三轮目标训练（Epoch 0/1/2）
只用高置信度路线，从第四轮（Epoch 3）起启用挽回。
设为 0 则从第一轮开启；`--no_radius_rescue` 可关闭挽回进行对照。
两条接纳路线不重叠，因此 Final_Accept_Num = Confidence_Accept_Num + Radius_Rescue_Num。

## 置信度阈值与源域距离阈值

恢复原 v8-02 的类别自适应置信度阈值，每轮只扫描目标训练集的 Teacher 预测，
不使用目标域真实标签。每个预测类别统计平均最大概率 m_c，
g 是有预测样本的各类 m_c 的平均值：
```text
center = min(max(threshold_base, g + threshold_margin), threshold_max)
tau_c = clamp(center + threshold_beta * (m_c - g), threshold_min, threshold_max)
```
没有预测样本的类别使用 threshold_max。
默认 base=0.85、beta=0.5、margin=0.02、min=0.80、max=0.95。

距离阈值继续采用源域真实标签分组、L2 归一化特征的均值原型 mu_c，
R_c 为最远 min(11, N_c) 个源域样本到 mu_c 的欧氏距离的均值。
目标样本用两弱视图归一化特征相加再归一化，只比较到其预测类别原型的距离。
类别缺失、非法特征不能通过距离判断；R_c=0 时严格小于条件不接纳。
无方差估计，无目标域近邻搜索或投票。

## 原型、角色和日志

最终接纳样本用于目标域分类损失和原型一致性损失。
角色分离启用时，anchor 需连续 promotion_epochs 轮标签一致、最终接纳且距离通过，
用于共享知识/MMD 更新；这是独立于分类学习的限制。
`--immediate_prototype_update` 继续允许所有最终接纳样本即时更新目标域原型，
不改变 anchor 晋升规则。用于几何判断的源域原型和 R_c 不受该更新直接影响。

新增日志：Confidence_Thresholds、Radius_Rescue_Enabled、Confidence_Accept_Num、
Rescue_Candidate_Num、Radius_Rescue_Num，以及两条路线各自的类别分布。
Radius_Pass_Num 是所有目标样本中的几何通过数，不等于挽回数。
旧 v8-03 文档描述的是直接距离筛选；本文件描述当前分支的 OR 挽回策略。

## 运行

```bash
git fetch origin
git switch --track origin/experiment/dual-view-confidence-radius-rescue-v8-04

mkdir -p test_los
set -o pipefail
python -u train.py \
  --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_source_radius11_direct_source_best.pth \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --data1 rafdb --data2 fer \
  --backbone mobilenet_v2 \
  --pre_epochs 0 --epochs 30 \
  --source_radius_k 11 \
  --radius_rescue_warmup_epochs 3 \
  --immediate_prototype_update \
  --run_name confidence_radius11_rescue \
  2>&1 | tee "test_los/confidence_radius11_rescue_$(date +%Y%m%d_%H%M%S).log"
```

在 CAST 仓库根目录执行。checkpoint 仅加载模型权重，重新建立 Teacher/原型/阈值，
不恢复目标训练优化器和历史状态。最佳目标模型：
`models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_target_best.pth`。

按用户要求，本次未新增或运行测试，未运行真实数据训练。
本次修改恢复指定的挽回语义，不保证准确率提升：开启挽回后，如果半径依然宽松，
大部分低置信度候选仍可能被接纳。对固定的 Teacher 和距离条件，OR 接纳集合包含
直接距离筛选集合；恢复高置信度路线本身并不会使筛选更严格。
