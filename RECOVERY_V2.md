# Prototype Recovery v2

分支：`experiment/temporal-consistency-v1`。本次根据用户补充的
`temporal_hard_only` 完整日志修正默认行为，替代 `6177e63` 的默认实验设置。

## 先区分事实和推断

| 指标 | 之前的原型实验 | 关闭软监督、保留时序硬筛选 |
| --- | ---: | ---: |
| 源域模型最佳目标验证准确率 | 51.07% | 51.07% |
| 适应后最佳验证准确率 | 57.56% | 56.70% |
| 按验证集选中的模型测试准确率 | 56.39% | 56.09% |
| 最后 epoch 接受伪标签数 | 19,396 | 17,149 |

验证差距是 0.86 个百分点；测试差距是 0.30 个百分点，3589 张图上约 11 张。
没有重复种子和逐样本预测，不能断言这个测试差距具有统计显著性，也不能把差距全部归因于一个模块。
这里比较的是用户提供的原型实验日志，不是论文表格中的结果；没有核验“与原文相同”的具体配置和指标。

新运行确实使用 `target_soft_weight:0.0`，但仍是 `temporal enabled:True`。
因此关闭软监督并未恢复旧原型基线。源模型验证分数一致，学习率轨迹也一致，
没有日志证据表明此次问题是源权重加载错误或学习率被重置。

最后一轮时序筛选的具体影响：

- 双视图候选 18,622，最终接受 17,149，剔除 1,473（约 7.91%）。
- `History_Disagrees` 只有 12；主要限制来自连续通过置信门限的要求。
- happy 保留率 95.89%，disgust 82.07%，neutral 89.17%，sad 88.60%。
- 这说明硬筛选改变了类别覆盖；仅凭日志不能证明被剔除的样本一定正确或错误。
- `Label_Flips:2995` 是所有目标样本的预测类别翻转数，不是 2,995 个已接受伪标签被纠错。

更深的弱项在进入目标阶段前已经存在：source checkpoint 的 fear 验证召回率只有 7.66%，
最终选中 student 的 fear 验证召回率为 8.47%。RAF-DB 的 fear 有 281 张，happy 有 4772 张。
这是尝试源域类别平衡的理由之一，但类别数量与低召回不能单独证明因果。
`class mean confidence` 只对“预测为该类”的样本求均值，高均值不能替代该类的召回率或准确率。

此外，v1 把 FER 文件列表改成先排序，且 source checkpoint 复用时未独立重置目标阶段随机起点。
从头训练 30 轮和跳过 source 训练会消耗不同的随机数；即使模型/optimizer/scheduler 相同，
batch 顺序、增强和 dropout 也可能不同。这使过去的单次对比不够严格。

## 修改内容

### 1. 默认恢复原型基线的直接监督

新的默认值：

```text
temporal_mode = observe
target_soft_weight = 0
source_balance_power = 0
```

`observe` 保存时序统计，但所有训练损失和原型更新都使用原来的双视图掩码。
这样可以继续收集诊断，同时不减少已通过原门限的目标监督。分类、特征对齐、原型损失共用这个掩码。
`--temporal_mode off` 完全关闭记忆；旧的 `--disable_temporal` 是其兼容别名。
只有显式 `--temporal_mode filter` 才启用原来的时序硬筛选。

soft CE 也改为显式开启。`observe` 搭配非零 `target_soft_weight` 仍会用历史概率做软监督；
因此只有 soft weight 为 0 时，才能把 observe 当作完全不影响优化的诊断模式。

基础 backbone、增强、阈值公式、EMA、原型设置、损失权重和 LR 调度未重新调参。
这恢复的是监督规则，不是历史运行的逐位复现承诺。

### 2. 固定目标阶段的实验条件

- 新增 `--seed`，默认 1314；进入 target 阶段时重新设置 Python、NumPy、PyTorch 的随机状态。
- source、target、threshold、prototype、validation、test DataLoader 各有独立 generator。
  目标阶段同时重置这些 generator，避免是否运行过 source 阶段改变后续采样顺序。
- worker 用自己的初始种子初始化 Python/NumPy 增强；数据集内部改为局部 RNG，不再重置全局 NumPy。
- RAF 图像和标签使用同一个局部 permutation，保持配对及原来的固定打乱策略。
- 日志记录输入 checkpoint SHA256、target 文件顺序 SHA256、seed 和实际 target 初始 LR。
  文件顺序哈希不验证图像内容；对比时仍需相同的数据、worker 数、软件和设备。

worker/generator 的设置参考 [PyTorch 可复现性说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html#dataloader)。
固定随机源也不保证不同 PyTorch/CUDA 版本、设备间逐位相同。

### 3. 独立可选：仅对有真值的 source CE 做温和类别加权

`--source_balance_power 0.5 --source_balance_max_ratio 2` 启用：

```text
w_c = min((max(source_counts) / source_counts[c]) ** power, max_ratio)
source_ce_i *= w_label_i / mean(w_labels_in_source_batch)
```

权重只从真实 source 原型初始化计数得到；不读取目标训练真值，不使用伪标签类别分布设权重。
权重的类别间比值最多 2，batch 内归一化保持 source 总样本权重不变，
target CE 和“source 样本数 + 已接受 target 数”的最终分母不变。
该开关仅作用于 target 阶段的 source CE，不重训源模型，也不更改特征对齐/原型目标。

给定本次 RAF 数量，类别顺序 `[surprise, fear, disgust, happy, sad, angry, neutral]`
的相对权重约为 `[1.9233, 2, 2, 1, 1.5517, 2, 1.3750]`。
这是一项待验证的实验，默认关闭；它可能改善弱类，也可能降低总准确率。

另外，日志现在分开报告 Source CE、Target CE 和逐类实际剔除数，避免只看混合 loss。

## 本地运行

先拉取当前分支，运行恢复后的基线；默认复用之前的 source_best，不重复预训练：

```bash
git switch experiment/temporal-consistency-v1
git pull --ff-only origin experiment/temporal-consistency-v1
bash scripts/run_recovery.sh --run_name recovery_base_s1314
```

默认 checkpoint 为：
`models/rafdb_fer/mobilenet_v2_rafdb_fer_prototype_consistency_v1_source_best.pth`。
路径不同则追加 `--checkpoint /your/source_best.pth`；数据路径仍可用 `--source_path`、`--target_path` 指定。
不要使用上一轮失败的 target checkpoint 当作此次 source 初始化。

随后如需验证弱类改进，仅改变 source 权重，其他条件与上面一致：

```bash
bash scripts/run_recovery.sh \
  --source_balance_power 0.5 \
  --source_balance_max_ratio 2 \
  --run_name recovery_source_balance_s1314
```

比较两组的 **best validation accuracy** 和逐类 validation recall，确认初始 checkpoint 哈希、文件顺序哈希、
seed、worker 数及 LR 一致。不要根据末尾 test 分数选择类别权重或下一组超参数。
若加权只改善宏平均召回却降低总体验证准确率，而目标仍是总体准确率，应保留恢复基线。
有稳定收益后，再以相同的 source checkpoint 对两组分别使用 `--seed 1315`、`--seed 1316` 复核，
避免根据单个最好结果判断增益。

旧脚本 `scripts/run_temporal_consistency.sh` 也已更新为恢复默认值，但它默认会运行完整 source 阶段；
想节省重复预训练请用上面的 `run_recovery.sh`。
要显式复查 v1 的筛选方案，添加 `--temporal_mode filter --target_soft_weight 0`。
要复查带软监督的 v1，添加 `--temporal_mode filter --target_soft_weight 0.5`。
它们采用新的随机设置，因此同样不应声称逐位复现旧日志。

## 验证与边界

```bash
python -m unittest discover -s tests -v
python -m py_compile train.py dataset.py temporal_utils.py training_utils.py
bash -n scripts/run_temporal_consistency.sh
bash -n scripts/run_recovery.sh
```

包含真实训练编排的 CPU 合成测试：observe/off 三轮权重一致，完整 source→target 与复用同一个 source checkpoint
的 target 权重一致；还覆盖权重上限、目标梯度不被源域加权改变、局部 RNG、各 loader 随机流隔离、
多 worker 的种子复现，以及已有时序/EMA/checkpoint 测试。多 worker 测试用 Python 标量返回增强结果，
不覆盖当前受限环境不支持的共享 Tensor 本地套接字传输。

本环境仍没有 RAF-DB/FER2013 数据或 GPU，未重跑真实 30 轮训练。
已修复默认行为和实验对比的缺陷；是否恢复或超过之前的 57.56% validation / 56.39% test，
仍需上述同条件实验确认。
