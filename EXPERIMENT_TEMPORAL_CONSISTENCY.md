# Temporal Consistency v1：日志诊断与本地实验

目标分支：`experiment/temporal-consistency-v1`。本次修改基于
`3b4c5d6825cca022564c77bcfaf3f65757f507b6`；该提交虽然处于 temporal 分支，
实际仍是 Prototype Consistency v1，没有按样本保存跨 epoch 预测。

## 日志说明了什么

以下数字来自本次提供的 MobileNetV2、RAF-DB → FER2013 运行日志，不是新代码的结果。

| 现象 | 日志证据 | 判断与优先方向 |
| --- | --- | --- |
| 仍有明显跨域差距 | 源域最后训练准确率 98.44%；源域阶段最佳目标验证准确率 51.07%；适应后最佳验证 57.56%，最终测试 56.39% | 后续可独立试验源域正则化、针对 FER 低分辨率/灰度特征的增强；本次保持 backbone 和增强不变 |
| 预测越来越自信，但收益减弱 | target epoch 1 → 29：分类损失 0.168 → 0.044，验证损失 1.632 → 2.349；最佳验证出现在 epoch 26，而最后只有 56.59% | 与过度自信/确认偏差相符，但日志不能证明哪些伪标签错误；本次加入历史筛选与软监督 |
| 原型指标已接近饱和 | epoch 0 原型一致率 99.81%，epoch 29 为 99.96% | 指标仅统计已被双视图接受的样本，不是伪标签准确率；直接提高原型权重或加一个同样的原型门限，未必提供有效新信息 |
| 接受的伪标签分布偏斜 | epoch 26 的 19,280 个伪标签中 fear 529（2.74%）、angry 1,108（5.75%）、happy 6,532（33.88%） | 先看逐类验证召回与筛选保留率，再独立试验温和的类别权重；分布偏斜本身不能证明错误，也不应强行改成均匀分布 |
| EMA 只参与生成标签 | 原代码每轮只验证 student，最终也只加载 student | 增加可选 `--eval_ema`，在验证集上比较 student/teacher，可能获得更合适的推理模型 |

本次实验借鉴 [Temporal Ensembling](https://arxiv.org/abs/1610.02242)
的样本预测历史思路，保留原来的 [Mean Teacher](https://arxiv.org/abs/1703.01780)
参数 EMA。这里的连续置信筛选和混合 CE 是本仓库的实验实现，并非论文的完整复现。

## 本次改动

1. **稳定样本索引**：FER 先排序文件再执行原来的固定种子 shuffle；训练时返回样本索引。
   同一个样本即使每轮 batch 顺序不同，也只更新自己的历史。
2. **时序筛选**：前 2 个 target epoch 保持原双视图掩码并收集历史；之后要求当前双视图均过原门限、
   连续至少 2 个 epoch 以同一类别通过双视图筛选，并且当前类别与更新前的历史概率 argmax 一致。
   低置信、类别翻转或漏掉一轮都会打断连续计数。筛选发生在写入本轮历史之前。
3. **软监督**：用两弱视图平均概率更新每个样本的历史（momentum 0.7）。预热后，
   目标 CE = 0.5 × 原 hard CE + 0.5 × 历史分布 soft CE。保留非首选类别的概率，
   不做 sharpening。源域仍用 hard CE，分类损失仍除以“源样本数 + 接受目标样本数”。
4. **一致的损失掩码**：时序筛选后的掩码同时用于目标 CE、CAST 特征对齐、原型损失和目标原型更新。
   其他 backbone、增强、损失权重、阈值公式和优化器调度保持原设置。
5. **可选 EMA 验证与模型选择**：`--eval_ema` 才额外验证 teacher；只用 validation 选择阶段、epoch、
   student/teacher。如果目标适应从未超过源域最佳验证分数，保留源域模型。最后对选中的模型测试一次。
6. **诊断与 checkpoint**：输出筛选前/后伪标签分布、逐类保留率、历史冲突数、标签翻转数、
   每类 validation/test 召回与宏平均召回。验证 loss 按样本平均，避免最后一个小 batch 权重过大。
   新 checkpoint 使用 `run_name`，保存 temporal bank、相对文件顺序和 `selected_model`。

这里“屏蔽”是指损失中的直接监督；与原实现相同，未接受的目标图像仍参与整个 batch 的前向传播，
因此仍可能影响 student 的 BatchNorm 统计。

时序稳定并不保证标签正确，也可能让低置信类别更难进入训练。历史会吸收所有观察，包括被拒绝的样本，
使早期错误有机会纠正。请结合 `Retention_By_Class` 和逐类验证召回判断，不以伪标签数量最大化为目标。
本次没有增加任何读取 FER 训练真值的优化项，也没有把日志中的类别真值分布设成训练先验。

## 推荐运行

先拉取此分支，在已有 `(cast)` 环境中执行：

```bash
git switch experiment/temporal-consistency-v1
git pull --ff-only origin experiment/temporal-consistency-v1
bash scripts/run_temporal_consistency.sh
```

脚本保留日志中的全部基础参数，启用时序筛选、0.5 软监督和 EMA 验证，自动创建日志目录，
并为日志和 checkpoint 使用带时间戳的名称。原有数据路径默认值保持不变，可以追加
`--source_path /your/raf-basic --target_path /your/fer2013`。

已有本次原型实验的最佳源域 checkpoint 时，可以跳过重复的 30 轮源域预训练：

```bash
bash scripts/run_temporal_consistency.sh \
  --pre_epochs 0 \
  --checkpoint models/rafdb_fer/mobilenet_v2_rafdb_fer_prototype_consistency_v1_source_best.pth
```

该文件路径来自旧版命名规则；请确认文件在本地存在。代码会加载该 source checkpoint 的模型、
optimizer、scheduler（后两项存在时），重新计算源模型 validation 分数，再从 target epoch 0 开始。
`--checkpoint` **不是完整目标阶段断点续训**：此模式拒绝包含 EMA teacher 的 target checkpoint。
temporal bank 和文件顺序虽被保存，当前 CLI 不恢复中途的 target 训练状态。

## 消融顺序

直接使用 `python train.py` 可单独控制 EMA 验证。下表各行追加同一个已确认存在的
`--checkpoint SOURCE_BEST.pth --pre_epochs 0 --epochs 30 --backbone mobilenet_v2`。
未列出的基础参数默认与提供的日志一致。

| 对比 | 追加参数 | 要回答的问题 |
| --- | --- | --- |
| A：原型基线 | `--disable_temporal --target_soft_weight 0 --run_name ablation_prototype` | 新的数据排序和评估修复下，重建对照 |
| B：仅时序筛选 | `--target_soft_weight 0 --run_name ablation_temporal_gate` | 过滤不稳定样本是否改善验证准确率 |
| C：时序 + 软监督 | `--target_soft_weight 0.5 --run_name ablation_temporal_soft` | 保留不确定性是否额外有益 |
| D：C + EMA 选择 | `--target_soft_weight 0.5 --eval_ema --run_name ablation_temporal_ema` | EMA 模型能否优于 student |
| 可选：仅软监督 | `--disable_temporal --target_soft_weight 0.5 --run_name ablation_soft_only` | 更严格筛选若损伤低置信类别，软监督是否仍有益 |

`--disable_temporal` 只关闭记忆和时序筛选；软监督若开启，使用当前双视图平均概率。
`--target_soft_weight 0` 才完全关闭软监督。两者分别可消融。软监督在
`--temporal_warmup_epochs` 后启用，即使时序记忆关闭也是如此。

所有对比都先依据验证准确率和逐类召回选配置；测试集结果只用于最终报告，不用于调整门限或选择下一组参数。
FER 文件排序的稳定化会改变旧运行的样本顺序，因此 A 是实现行为上的基线，不承诺逐位复现上传日志。

如继续研究类别偏置，下一步应在 C/B 中验证更好的配置上，单独加入带上限的弱类别重加权，
并检查 fear/angry 召回及总准确率是否同时改善。源域正则化和 FER 风格增强也应独立实验，
避免多个机制同时变化后无法归因。

## 读取最终选中的权重

target checkpoint 的 `model` 仍保存 student，以对应 optimizer 状态；如果
`selected_model == 'ema_teacher'`，推理应加载 `ema_teacher`。训练脚本最后的 test 已正确处理此选择。

```python
checkpoint = torch.load(checkpoint_path, map_location='cpu')
key = 'ema_teacher' if checkpoint.get('selected_model') == 'ema_teacher' else 'model'
model.load_state_dict(checkpoint[key])
model.eval()
```

如果末尾输出 `selected model source_student`，应使用本次运行的 `_source_best.pth`。

## 已完成的验证与边界

```bash
python -m py_compile train.py dataset.py temporal_utils.py
python -m unittest discover -s tests -v
bash -n scripts/run_temporal_consistency.sh
```

测试覆盖时序历史防止本轮自我验证、打乱顺序、连续计数重置、错误恢复、双视图约束、checkpoint、
软标签梯度隔离、空掩码和数据索引；微型合成数据跑过真实训练编排，覆盖 source/target、
EMA 更新、原型损失、三种模型选择以及源 checkpoint 复用，并用非法 target 训练标签检查监督泄漏。
另做了真实 MobileNetV2 的 CPU 前向/反向检查。

执行环境为 PyTorch 2.5.1 CPU；没有用户的 RAF-DB/FER2013 数据与 GPU，未完成新的 30+30 轮训练。
兼容旧版 PyTorch 的 soft CE 使用手写 log-softmax，但未在仓库 README 的 PyTorch 1.8.1 上实测。
**本提交提供可运行、可消融的改进实验，不宣称已超过 57.56% validation / 56.39% test。**
