# CAST

This is a PyTorch implementation of the paper:

"Unsupervised Cross-Domain Facial Expression Recognition via Class Adaptive self-Training"
## Environment
Ubuntu 16.04 LTS, python 3.8, pytorch 1.8.1
## Datasets
[RAFDB](http://www.whdeng.cn/raf/model1.html),
[AFE](https://github.com/HCPLab-SYSU/CD-FER-Benchmark),
[EXPW](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html),
[SFEW](https://paperswithcode.com/dataset/sfew),
[FER2013](https://paperswithcode.com/dataset/fer2013)


## RAF-DB → FER2013: 稳定适配与复现

2026-09-12 的修复及运行结果分析见 [诊断说明](docs/2026-09-12-training-diagnosis.md)。
本次修改已完成 CPU 回归测试，尚未在真实 RAF/FER 数据上验证准确率提升。

优先复用你这次日志中已经保存的 source checkpoint，保持适配起点一致：

```bash
git pull --ff-only
bash scripts/run_fer_stable.sh
```

脚本默认读取 `./models/rafdb_fer/resnet50_rafdb_fer_source_best.pth`。该权重在你的训练机器上，
不会随 GitHub 下载；换路径时用：

```bash
CAST_SOURCE_CHECKPOINT=/path/to/source_best.pth bash scripts/run_fer_stable.sh
```

如果没有 source checkpoint，从头训练：

```bash
python -u train.py --backbone resnet50 --source_epochs 30 --epochs 30 \
  --w2 0 --target_w2 0 --w3 0.1 --target_lr 0.0003 --ema_decay 0.995
```

数据路径仍默认使用原训练机器路径，可通过 `--source_root` 和 `--target_root` 修改。
`--checkpoint` 仅初始化权重后继续训练源域；新增的 `--source_checkpoint` 才会跳过源域训练。
目标域重新初始化 Adam 和学习率调度器，不继承 source 最佳轮次的隐含学习率。

每次结果保存到 `models/rafdb_fer/<run_name>/`，同名目录会报错，避免覆盖旧实验：

- `source_best.pth`：适配起点。
- `target_student_best.pth` / `target_ema_best.pth`：分别保存最佳学生与 EMA 模型；若适配未提升则保留源域起点。
- `best.pth`：两者及源域起点中的最佳验证候选，`model_kind` 标明来源。
- `config.json`：完整参数、Git 版本及脏状态、框架版本、初始权重 SHA256、数据路径/标签清单摘要。
- `history.jsonl`：逐轮学生/EMA 准确率、逐类召回率、混淆矩阵、伪标签分布、独立源/目标分类损失。
- `summary.json`：两种模型的最佳结果。

checkpoint 的 `model` 字段可以直接用于推理；同时保存优化器、调度器和目标阶段的学生/教师权重。
目前没有完整的中断续训功能，加载 `--source_checkpoint` 表示开始一次新的适配实验。

### 对照实验

使用同一个 source checkpoint、seed、数据与训练轮数，分别运行：

```bash
# 稳定配置：先检查关闭目标 affinity 后的伪标签质量
bash scripts/run_fer_stable.sh
# 只打开目标 affinity，比较它在修复后的训练中是否有益
bash scripts/run_fer_stable.sh --target_w2 0.03
# 只恢复原先的教师增强，检查强增强对伪标签的影响
bash scripts/run_fer_stable.sh --teacher_views legacy
# 只恢复旧的分类损失归约
bash scripts/run_fer_stable.sh --target_loss_reduction legacy
```

`legacy` 选项只控制对应的一项机制，不会完整复原旧提交。不要把多处修改之间的差值
解释为某一项机制的因果效果。与旧日志直接比较时先看 **student best**；EMA best 是新增候选。

### 验证

测试使用合成图片及 CPU，不需要真实数据或预训练权重：

```bash
python -m pytest tests -q
```

本次验证环境：Python 3.12、torch 2.6.0+cpu、torchvision 0.21.0+cpu、
OpenCV 4.11.0、pytest 8.3.5。代码未要求升级你的现有 CUDA/PyTorch 环境；
原 README 的 PyTorch 1.8.1 环境未在本次重新验证。
