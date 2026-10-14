# 有角色权限的双视图邻域关系约束（实验）
基于 v8-05 03378311。没有历史组排除、子原型或邻域标签投票。--neighbor_structure 只新增结构KL损失，保持高置信度OR半径救回、连续anchor晋升、原型一致性和MMD。必须开启角色分离且不启用 immediate_prototype_update。

## 机制
上一轮anchor的融合Teacher特征构成每轮冻结库，默认最多3000个唯一ID，稳定ID哈希采样。其他接纳样本只能作为结构接收者，不能立即进入库；不接纳样本不接收结构损失。查询自身被排除。
当前Teacher双弱视图归一化均值选top-k近邻（默认10），相似度>=.5，并要求反向近邻条件：候选相对查询的相似度不低于该候选库内第k个邻居的相似度。已有库查询用它的冻结特征判断反向关系，新查询使用当前融合特征模拟插入。并列允许；这是含陈旧特征的互近邻判据，不保证两个当前视图的严格双向kNN。
剩余邻居至少3个。两个Teacher视图分别在同一参考集合上计算 softmax(cos/.2)，Jensen-Shannon散度<=.05才允许传递。两分布均值为目标q。Student强视图在固定参考向量上计算分布p，损失KL(q||p)，仅Student查询特征有梯度。目标不使用任何邻居标签。对有资格查询求均值；没有资格返回可微零。
结构权重默认.05，warmup3轮、ramp5轮，沿用 prototype_weight_for_epoch 调度。它约束近邻相对关系而非类别概率或分类器输出。若局部分布近乎均匀，损失可能非常弱；稳定错误结构也可能被保持，不能保证去噪。

## 日志
Neighbor_Structure 的bank是实际冻结库大小，eligible是最终接纳查询数，used是符合互近邻及JS条件的查询数，loss是未加权结构损失的batch平均，weight是当前权重。used长期为0则没有结构监督，loss极小也不等于改善。
CPU缓存是上一轮不同batch Teacher特征，存在陈旧性；CPU检索增加开销。checkpoint保存库，但 --checkpoint 仍仅初始化Student权重，不是完整恢复。

## 运行
```bash
python -u train.py \
 --checkpoint /workspace/ttt/code/CAST/models/rafdb_fer/mobilenet_v2_rafdb_fer_confidence_radius11_rescue_source_best.pth \
 --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
 --target_path /workspace/ttt/code/data/fer2013 \
 --backbone mobilenet_v2 --pre_epochs 0 --epochs 30 \
 --source_radius_quantile .90 --source_refresh_interval 1 \
 --neighbor_structure --structure_weight .05 --structure_k 10 \
 --structure_min_neighbors 3 --structure_temperature .2 --structure_max_js .05 \
 --run_name neighbor_structure_anchor
```
不启用 --neighbor_structure 即为A对照，均不加 --immediate_prototype_update/--no_role_separation。正式评估需同初始化多随机种子及结构数量/稳定性消融。
验证：5个NumPy测试覆盖自排除、概率归一化、双视图分歧拒绝、未接纳查询拒绝、空库/零特征及确定性容量采样；CLI与语法检查通过。环境没有PyTorch/训练数据，未运行实际KL反向传播或端到端训练，不宣称准确率提高或首创。
