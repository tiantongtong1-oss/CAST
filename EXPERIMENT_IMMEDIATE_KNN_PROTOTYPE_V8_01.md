# v8-01：KNN 新救回样本立即更新目标原型

基于 experiment/role-separated-sparse-knn-v8（4d4616fda0d9cba3626bed6bf45f911a99c550df）。
新分支：experiment/role-separated-sparse-knn-v8-01。

## 实验开关

新增 --immediate_knn_prototype_update，默认关闭，便于在同一份代码上比较：

- 不加开关：原始 v8，只有晋升锚点更新目标原型。
- 加开关：更新集合为 anchor_mask OR rescue_mask。新救回样本在当前 batch 的优化器与 EMA 更新之后，用本次前向得到的双弱视图 Teacher 特征更新目标原型；无需等待晋升。更新后的原型从后续 batch 起影响原型一致性学习；预热期间仍可更新原型，原型损失沿用原有预热与渐增。
- rescue_mask 是未通过高置信度通道、双视图一致且通过 KNN 救回的样本。非锚点高置信度样本不会因此获得即时原型写入资格。
- 已经晋升且又被救回的样本只统计和更新一次。
- MMD 仍使用 anchor_mask，邻域发送仍遵循原来的晋升和刷新规则。软标签接收者不加入更新集合。
- 高斯/邻域门槛、分类损失、原型动量与源原型混合、模型选择协议均沿用 v8。
- 不要同时加 --no_role_separation；该开关会同时改变 MMD 和发送者资格，失去单因素比较意义。

“立即”指本 batch 末写入原型，并非回头改变本 batch 已经计算的损失。

## 修改文件

| 文件 | 内容 |
| --- | --- |
| train.py | CLI 开关、独立原型写入掩码、更新统计；checkpoint 的 args 自动记录开关 |
| tests/test_immediate_knn_prototype.py | 合成数据训练测试：开关、KNN 预热、原型更新、MMD 和晋升隔离 |
| EXPERIMENT_IMMEDIATE_KNN_PROTOTYPE_V8_01.md | 运行与比较说明 |

## 获取与运行

已有仓库中执行：

```bash
git fetch origin
git switch --track origin/experiment/role-separated-sparse-knn-v8-01
```

使用同一个源域 checkpoint、数据划分、随机种子和训练预算。将下面占位路径换成实际位置，沿用原实验其余参数：

```bash
python -u train.py \
  --checkpoint /path/to/source_best.pth \
  --source_path /path/to/raf-basic \
  --target_path /path/to/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 0 --epochs 30 \
  --immediate_knn_prototype_update \
  --run_name role_sparse_v8_01_immediate
```

对照组删除 --immediate_knn_prototype_update，并改为 --run_name role_sparse_v8_01_control。
--checkpoint 是权重初始化，不是完整训练恢复。不要用已经做过目标域适应的 checkpoint 当源域初始化。

## 日志与判断

- Prototype_Update_Num：本轮参与原型更新的样本观测总数。
- Immediate_Rescue_Update_Num：其中尚未晋升、因本开关新增的 KNN 救回样本数。
- Anchor_Num 和 Stable_Senders：仍是原有锚点与发送资格统计。
- Target_Prototype_Counts：累计实际原型更新计数。

若 Immediate_Rescue_Update_Num 始终为 0，这项实验没有新增原型写入者，不能据此评判该机制的有效性。
优先比较 UAR、Macro-F1、Fear/Disgust 召回及整体准确率，不以原型更新数量增长代替性能提升。
此分支仍按目标域有标签 validation accuracy 选 checkpoint，应如实披露；此次只测试原型更新资格。

测试使用合成数据与小模型，并控制置信度/KNN gate 输出以覆盖指定路径；没有进行真实 RAF→FER 训练，不宣称准确率提升。
