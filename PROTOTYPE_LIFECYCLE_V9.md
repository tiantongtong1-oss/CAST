# 互为近邻群体驱动的子原型生命周期（实验）
基于 v8-05 03378311。保持 final_mask = confidence OR radius rescue、源域分位数半径、连续 anchor 晋升、MMD 角色分离；禁止启用该功能时关闭角色分离或开启所有接纳样本即时写入。

## 每轮执行
1. 从上一轮目标 anchor 收集唯一 ID、Teacher 融合特征、伪标签。固定 ID%5==0 作为外部审核池，其他 ID 构建群体；审核池不参加图、群体均值或成员持续性匹配。它们仍参与原 A 的普通训练及父原型，故这是对子原型构建的外部审核，不是对整个模型统计独立。
2. 在所有构建 anchor 上先建立余弦互为 top-k 近邻图（不按标签分图），最低边相似度默认 .8。取连通分量，丢弃小分量。为控制开销，图最多 3000 个 ID，采用稳定 ID 哈希采样；不是无损全数据图。
3. 图形成后要求群体伪标签纯度 >=.9，以多数类别赋语义，仅该类成员构建中心且至少20个不同ID。中心相对刷新源域类别均值的余弦优势至少 .02。源域均值在这里归一化用于语义审核；原半径计算不改变。
4. 通过同类成员 Jaccard>=.5 一对一跟踪，连续3轮才成为临时候选。一个类别至少存在两个持续群体才允许分裂。
5. 固定审核池该类至少10个样本；至少10个外部样本比父原型更接近候选，且引入候选后类别间隔平均改善 >=.005。其他每个审核伪类别的平均间隔不得下降超过 .01。只使用伪标签，不能保证真实标签正确。
6. 候选顺序按连续轮数、成员数、类别；审核在已经接纳的子原型基础上逐个进行。同类余弦>=.98的候选合并为一个代表（不做均值融合）；每类最多3个。每轮重建正式集合，消失、不持续或审核失败的候选立即撤销，没有宽限期。
7. 正式子原型仅增强 prototype consistency：每类 logit 为父原型及该类子原型的最大余弦相似度/温度；仍按伪类别平衡CE。分类器预测、源域距离门槛和 MMD 不使用子原型。基线原型诊断使用同一 max 规则。

## 角色与历史
所有构建和审核均来源于上一轮 anchor，不接纳普通低置信度样本写子原型。普通父原型继续仅 anchor 更新。epoch0为空，anchor晋升后再积累群体稳定性；启用时间由数据决定。目标特征是上一轮不同batch Teacher缓存，存在陈旧性；群体ID轨迹不等于标签正确性或真实模式。连通分量可能合并不同模式，也可能碎片化，本功能不保证产生子原型。
子原型 buffer及生命周期轨迹保存到 checkpoint；原入口仍仅加载 Student 权重，不新增完整恢复。图CPU计算增加耗时。可去掉 --prototype_lifecycle 作为A对照，父原型损失在无子原型时保持原计算。

## 命令
```bash
python -u train.py \
 --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
 --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
 --target_path /workspace/ttt/code/data/fer2013 \
 --backbone mobilenet_v2 --pre_epochs 0 --epochs 30 \
 --source_radius_quantile .90 --source_refresh_interval 1 \
 --prototype_lifecycle --community_k 10 --community_similarity .8 \
 --community_min_size 20 --community_persistence 3 \
 --child_audit_min 10 --child_max_per_class 3 \
 --run_name mutual_graph_lifecycle_anchor
```
不要添加 --immediate_prototype_update 或 --no_role_separation。日志 Prototype_Lifecycle 展示 communities/tracked/persistent/active/merged/audit_failed。active长期为0即没有正式启用，不能宣称新机制改善结果。

## 验证限制
NumPy 图/生命周期测试覆盖几何分组、延迟激活、群体消失撤销、外部无收益拒绝、源语义拒绝、审核ID不影响中心、重复ID。另做CLI/语法检查。当前环境无PyTorch/FER，未运行实际反向传播和端到端训练，不保证性能提升或首创。
