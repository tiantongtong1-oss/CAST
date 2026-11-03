# Top-seven radius OR rescue

每轮用当前 EMA Teacher 在确定性源域视图提取归一化特征，按真实标签刷新归一化类别原型、每类最远 7 个欧氏距离的均值 R_c，以及真实标签源域邻居库。不使用目标训练集真实标签。

最终接纳 = 双弱视图一致 AND (高置信度 OR 半径补救 OR 邻居补救)。高置信度样本无需半径/邻居审核。

默认 epoch 索引 >= 3（日志 Epoch 3，即第4轮）开始低置信度补救：
1. 距离严格小于 0.7 R_c，直接补救。
2. 未通过半径，查询全类别源域最近 11 个邻居；严格多数真实标签等于 Teacher 原伪标签则补救。不改写伪标签。
3. 源域有效样本不足11时使用全部可用邻居，至少3个才可审核；严格多数为超过实际邻居数的一半。满11个要求至少6票。

--rescue_warmup_epochs 默认3，可显式设置。每类不足7个有效样本时半径补救关闭，但仍可使用全源域邻居审核。非有限/零特征拒绝补救。原型和半径每轮更新后在该轮固定。

日志自动保存在 logs/，逐轮记录 Radius_Rescue_Num、Neighbor_Rescue_Num、Neighbor_Checked_Num 和 Final_Accept_Num。权重使用 _prototype_radius_top7_07_or_ 后缀，与旧 AND 实验区分。

运行沿用 python train.py 参数或 scripts/run_prototype_radius.sh。测试：python -m unittest discover -s tests -p 'test_*radius.py'；python -m unittest discover -s tests -p test_or_rescue.py。
本次语法检查通过；执行环境缺少 PyTorch，数值测试和完整训练尚未运行。旧 Gaussian/KNN integration 测试仅为历史接口测试。
