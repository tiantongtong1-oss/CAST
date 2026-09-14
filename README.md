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

## `new` 分支 v5：按运行退化日志修复

运行入口仍是 **`train.py`**。上一版 v4 的实际最佳结果为 50.71%，最终 EMA 为 46.61%；
本轮未运行测试、训练或准确率评估，不声明已达到 65%。

### 复用本次日志中已保存的源域模型

在仓库目录运行：

```bash
git switch new
git pull --ff-only origin new
CUDA_VISIBLE_DEVICES=0 bash scripts/run_fer_v5.sh
```

脚本读取日志中明确保存的
`./models/cast_resnet50_v4/resnet50_rafdb_fer_source_final.pth`。
它必须是 **ResNet50 源域模型**，而不是已经退化的目标域 final 模型。
路径不同可以这样运行：

```bash
CAST_SOURCE_CHECKPOINT=/actual/path/source.pth bash scripts/run_fer_v5.sh
```

每次在 `models/cast_resnet50_v5/日期时间_进程号/` 新建输出目录，保存 `train.log`、
`config.json` 和 checkpoint；不会覆盖 v4。可追加 `--batch_size 32` 降低显存需求。
数据目录可通过 `CAST_SOURCE_ROOT`、`CAST_TARGET_ROOT` 修改。
本次新增源域验证集读取，需要 RAF 的 `test_*` 标注和对应 aligned 图片。

如需重新训练源模型，必须明确执行：

```bash
python -u train.py --backbone resnet50 --train_source --model_dir ./models/cast_v5_from_source
```

省略 checkpoint 不再悄悄触发 30 轮源域训练。新源域训练使用标签平滑，并按 RAF 源域验证
准确率保存 `*_source_best.pth` 作为适配起点；同时保留最后一轮源模型。

### 根据日志修改了什么

| 日志现象 | v5 改动 |
| --- | --- |
| 七类阈值始终等于 0.9 | 用源域验证集 NLL 自动选择温度；分位数阈值上界改为 0.995，保留类别间差异 |
| EMA 与源模型共享偏差，disgust 伪标签后期准确率约 12.8% | 使用去中心化的每类 3 个原型；不再通过源分类器同意来绕过原型检查 |
| 多个类别都被截为 1,200 个样本，低质量类别快速扩张 | 配额取决于目标训练图像的原型投票；跨轮增长上限默认 1.3 倍加 32；默认不进行逆频率加权 |
| 源域交叉熵后期由约 0.03 反弹到超过 2 | 源域和目标域各自维护 BN 统计；每步用弱增强目标图像更新目标统计，强增强不更新统计 |
| 硬伪标签可能强化过度自信 | 改为温度一致的软标签 KL，保留样本可靠性权重 |
| 使用不可靠类别补样、后续开启目标 DDRL | 默认关闭类别恢复和目标 DDRL；保留单项开启选项供消融 |
| 旧 MMD 包含自配对、随机丢弃较大类别的样本 | 按论文 Eq. (9) 排除对角线，保留不等长样本；类别损失加入 Eq. (12) 的 1/C 因子 |

这些是针对日志的实现与策略修改，日志不能证明每一项是退化的独立原因。
BN 每步统计更新会增加一次弱视图前向，运行时间取决于硬件；源原型仅初始化一次。

### 消融与兼容性

- `--bn_mode frozen`：使用 v4 的固定 BN 统计路径。
- `--teacher_temperature 1`：关闭自动温度校准；默认 `0` 表示自动校准。
- `--prototype_centers 1`：每类仅使用一个去中心化原型。
- `--target_w2 0.03`：打开按可信样本筛选的目标 DDRL，默认值为 0。
- `--recovery_topk_per_class 64`：打开类别恢复，默认值为 0。
- `--class_reweight_power 0.5`：打开平方根逆频率重加权，默认值为 0。

上述选项各自只改变一项，并不完整复原旧版。
`--class_balance_factor` 已弃用，以 `--pseudo_keep_fraction` 和 `--class_growth_factor` 控制配额。
`train_v3.py` 是旧训练入口，不包含 v5 的训练流程。

checkpoint 的 `model` 字段导出普通 CAST 权重和目标 BN 统计，可用原 `Networks.Model`
严格加载进行目标域推理；源域专用 BN 缓冲区仅用于训练，不写入该推理字段。
`--checkpoint` 初始化一个新的适配实验，未实现精确断点续训。

温度校准仅用源域验证标签；伪标签、配额、BN 更新和训练损失不使用目标真实标签。
目标训练标签只用于既有诊断打印。目标评估沿用此前每轮在 `test/` 上选最佳模型的流程，
`*_best.pth` 的分数应称为该集合上的验证选模分数；`*_final.pth` 是预设轮数的最终模型。

本轮已根据用户提供的论文全文核对 Eq. (9)、Eq. (12) 和训练设置。
原 CAST 已有类别自适应阈值及类条件对齐；分域 BN、温度校准、软标签训练也都有既有研究。
本项目的改动不等同于证明这些机制首创，方法有效性需要同一起点的消融实验。

参考：[CAST 论文](https://ieeexplore.ieee.org/document/10843182)、
[原作者实现](https://github.com/smwanghhh/CAST)、
[Domain-Specific Batch Normalization, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.html)。
