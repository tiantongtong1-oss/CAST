# Per-class prototype radius experiment

本分支替换 Global Gaussian / KNN 筛选，使用 7 个类别各自的固定源域原型和独立初始半径。

- 使用已有 `PrototypeBank.source_prototypes`，不另建均值中心；原型和样本均为 L2 归一化特征。
- 初始化 teacher 后，在确定性源域视图上再次扫描，以真实标签归类。
- 每类计算样本到该类原型的欧氏距离，取最大的 7 个距离的均值作为 R_c。流式保留 top 7，不保存全部源域特征。
- 原型和半径在目标训练期间固定，不做方差 EMA 或逐 epoch 半径刷新。
- 目标样本使用两个弱视图归一化特征之和的归一化结果；只检查预测伪标签对应的类别原型。
- 最终接收 = 双视图一致 AND 自适应置信度通过 AND 距离严格小于 0.7 R_c。从第一个目标 epoch 生效，无 OR 补救。
- 某类少于 7 个有效源域样本时，该类不接收伪标签；零半径按严格不等式也不接收。非有限值、零向量和非法标签均拒绝。
- 旧 `--knn_*` / `--no_knn_gate` 参数已移除；原 Gaussian/KNN 模块保留用于历史实验，不参与当前训练。旧 KNN 日志统计槽保留为零，Radius_Pass_Num 为半径通过数量。
- 权重文件使用 `_prototype_radius_top7_07_` 后缀，与 main 实验区分。

运行方式仍为 `bash scripts/run_prototype_radius.sh RAF_ROOT FER_ROOT`。

新数值测试：`python -m unittest discover -s tests -p test_prototype_radius.py -v`。
本次完成全仓库 Python 语法检查；执行环境未安装 PyTorch，因此数值测试未能运行，也未进行 GPU/数据集训练。旧 integration 测试面向 main 的高斯/KNN 接口，并非本分支训练入口的验证。
