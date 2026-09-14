# MobileNetV2 + Dual-View EMA CAST

分支：`improve/ema-dualview-cast`。本次基于 `913a754` 的代码继续改进。

## 本次修改与原有实现

原分支已经具备 MobileNetV2、双弱视图 EMA、全目标训练集统计、置信度加权 CE、MK-MMD DDRL、原型记忆和亲和损失调度。本次修复它们之间的实际训练路径，不将已有功能重复宣称为新增。

| 文件 / 位置 | 本次改动 | 作用 |
| --- | --- | --- |
| `Networks.py / build_backbone, Model` | 默认 MobileNetV2；兼容新旧 torchvision 权重 API，明确使用 ImageNet V1；推理特征留在模型所在设备 | 保持 1280 → 512 投影及已有 checkpoint 的参数键，避免教师特征反复传到 CPU |
| `Networks.py / compute_kernel_matrix` | 用 `torch.cdist` 替代显式 N × N × D 广播差值 | 避免巨大的三维距离中间张量；核矩阵仍为 N × N |
| `Networks.py / Model.forward(target)` | 先按原始 batch 边界分源/目标，再应用可靠掩码；按有效类别数归一化 | 部分源样本被屏蔽时仍正确分域，缺失类不稀释已有约束 |
| `ema_utils.py / PrototypeMemory.loss` | 用学生特征与正/负类原型计算吸引项和相对余弦 margin 分离项 | 原实现的原型-原型分离项无梯度；现在分离项能更新学生 backbone |
| `ema_utils.py / PrototypeMemory.update` | 拒绝零权重、零中心和非有限特征更新 | 防止无效原型被标记为已初始化 |
| `ema_utils.py / select_dual_view_pseudo_labels` | 回退也要求两个视图各自达到高置信度；没有合格样本时保持空掩码 | 避免单个过度自信视图掩盖另一个不确定视图 |
| `train.py / calculate_teacher_statistics` | 全集统计与 batch 筛选都先逐视图校正，再平均 | 修复两条路径中分布校正顺序不一致 |
| `train.py / target stage` | 源原型使用教师源特征，目标原型使用两弱视图归一化教师特征；学生强视图计算亲和损失 | 原型记忆关闭 dropout、无梯度，监督仍作用在学生 |
| `ema_utils.py / weighted_mean_loss` | 集中实现有效权重归一化，空掩码零梯度 | 统一目标 CE 与亲和损失的权重语义 |
| `train.py / run_training` | 目标阶段重新创建优化器/调度器，增加 `--target_lr`；修复 `--pre_epochs 0 --checkpoint` | 目标适应不继承源域衰减学习率和 Adam 动量；不误读旧的 source-best 文件 |
| `train.py / backward_and_step, parse_args` | 非有限 loss/梯度在 step 前报错；增加设备、batch、离线权重开关与参数检查 | 避免把坏梯度写入模型；支持 CPU 验证 |
| `tests/test_cast.py` | 数值、梯度、筛选和真实 MobileNetV2 两阶段合成数据测试 | 可复现地检查以上行为 |

`dataset.py` 的 FER2013 → CAST 映射及三视图接口保持现有实现。目标训练标签在伪标签生成、全局统计和损失中被丢弃。目标 val 标签仍用于选择最佳 checkpoint；这是沿用的实验设置，不应描述为完全不使用目标验证标签的模型选择。test 仅用于最终评估。

## 与框架图一致的实际计算

学生特征为 `MobileNetV2.features → GAP → Flatten → Dropout → Linear(1280,512) → Dropout`，分类器为 `Linear(512,7,bias=False) → BN(7)`。源图像与目标强增强图像拼接后经过同一个学生；教师是整个学生的 EMA 副本，独立保存参数且关闭梯度。

DDRL 是训练中的特征约束，不是额外的前向特征变换层。域对齐实现为类别条件 MK-MMD，没有额外的对抗判别器。类别结构使用类内集合与其余类别集合的负 MK-MMD。两项均使用有界的类别密度权重。

完整目标阶段损失（保留原 CAST 的分类器调制项）：

```text
L_cls = mean(CE_source) + lambda_t * weighted_mean(CE_target, pseudo_weights)
L_DDRL = L_align + L_sep
L_aff(i) = 1 - cos(f_i, p_y)
         + mean_{c != y, initialized} relu(cos(f_i,p_c) - cos(f_i,p_y) + margin)
L = w1 * L_cls + w2 * L_DDRL + lambda_aff(epoch) * L_aff + w3 * L_mod
```

`L_aff` 仅对可靠且正类原型已初始化的目标样本计算，并按其伪标签权重归一化。没有负类原型时分离项为零。原型是固定参照，学生特征接收梯度。默认亲和项前 2 个目标 epoch 权重为 0，之后 5 个 epoch 线性升至 0.1；目标 CE 默认 5 个 epoch 升至 1.0。EMA 在学生优化器成功 step 后更新。

原图的总损失省略了 `w3 * L_mod`；本分支保留这项，框架图会如实展示。图中的“增强特征”对应被 DDRL 损失约束的 512 维表示，不增加不存在的网络层。

## 运行

依赖 PyTorch、匹配的 torchvision、NumPy、Pandas、OpenCV、scikit-learn、Matplotlib、Pillow。原仓库记录的 PyTorch 1.8.1 环境未在本次重新验证；权重构造保留旧 torchvision API 兼容分支。新 API 依据 [torchvision MobileNetV2 文档](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html)，距离计算依据 [torch.cdist 文档](https://docs.pytorch.org/docs/main/generated/torch.cdist.html)。

```bash
python train.py \
  --source_path /path/to/raf-basic \
  --target_path /path/to/fer2013 \
  --backbone mobilenet_v2 --device cuda \
  --pre_epochs 30 --epochs 30 \
  --lr 0.001 --target_lr 0.001 \
  --w1 4 --w2 0.3 --w3 0.1
```

仅从已有、结构匹配的源模型开始目标适应：

```bash
python train.py \
  --source_path /path/to/raf-basic --target_path /path/to/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 0 --epochs 30 \
  --checkpoint /path/to/source_best.pth --target_lr 0.001
```

`--checkpoint` 加载模型权重，不表示恢复中断的目标训练状态；目标 EMA/原型记忆会重新初始化。目标 checkpoint 包含模型、教师、原型、优化器、调度器和运行参数。源/目标最佳 checkpoint 分开保存。

本地行为与合成数据验证：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

测试使用 `pretrained=False`，不下载模型权重，也不需要真实 RAF-DB/FER2013。合成图像只验证训练流程，不代表准确率或创新效果。正式实验建议比较原分支、亲和修复、教师原型、完整改进，并保持数据划分/种子一致。

## 本次验证结果

2026-09-14：Python 3.12、PyTorch 2.14.0+cpu、torchvision 0.29.0+cpu 下，15 项测试全部通过（60.770 秒），包含真实 MobileNetV2 的源预训练 + 目标适应，以及从指定 source checkpoint 启动目标训练。`git diff --check` 通过。当前工作区没有真实数据集和 GPU，未运行完整 RAF-DB → FER2013 训练，未报告准确率提升。
