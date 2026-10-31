# 类别平衡的均值加标准差公共半径（v30）

基于 experiment/bidirectional-balanced-common-radius-v29-run-history，
提交 c01ec162c25804f21d612f544eb57deedde01138。

每次源域刷新使用当前 EMA Teacher 提取 L2 归一化特征，各类均值原型保持独立，
原型均值不再归一化。计算源样本到其真实类别原型的欧氏距离 d。
每个非空类别总权重相同，类内每个有效样本权重为 1 / (C * n_c)。

- mu = sum(w * d)
- sigma = sqrt(sum(w * (d - mu)^2))，使用总体标准差，包含类内和类间距离波动。
- R = mu + k * sigma，所有类别共享 R。

新增参数 `--source_radius_std_scale`，默认 k=1.0，要求有限且非负。
旧参数 `--source_radius_quantile` 已移除；运行原 v29 命令时，将
`--source_radius_quantile 0.90` 替换为 `--source_radius_std_scale 1.0`。
它与置信度阈值的 `--threshold_std_scale` 是两个独立参数。

目标判断继续使用严格的 distance < R，覆盖率不再以90%为目标。
缺失类别不参与统计，且其几何判断始终拒绝；全空刷新时 R、mu、sigma 均为0。
不对 R 做额外截断或跨轮平滑。默认每个目标 epoch 刷新一次源域统计。

日志输出 Radius_Mode: class_balanced_mean_std、Std_Scale、Distance_Mean、
Distance_Std、Common_Radius 以及各类和类别平衡的实际严格覆盖率。
class_radii 的 checkpoint buffer 结构保持不变，各元素保存同一个 R。
模型检查点仍只用于权重初始化，不是完整训练恢复。

公共半径同时影响救援的 radius 分支和稳定 anchor 的资格判断。
置信度直接路径、KNN投票、竞争检查、稳定晋升和损失沿用v29逻辑。

验证：

```bash
python -m unittest discover -s tests -p 'test_balanced*radius.py' -v
python -m unittest discover -s tests -p test_global_mean_std_ema.py -v
python -m unittest discover -s tests -p test_neighbor_vote.py -v
python -m unittest discover -s tests -p test_competition_gate.py -v
python -m py_compile balanced_radius.py source_prototype_radius.py train.py
```

本次环境未安装 PyTorch，半径库集成测试自动跳过，未执行真实GPU训练。
历史 test_agreement_radius_ds.py 在原始v29和本分支均为1个失败、4个错误：
旧测试截取训练片段缺少 rescue_candidates，且仍断言旧的置信度上下限。
这些既有问题不属于本次半径算法修改范围。
