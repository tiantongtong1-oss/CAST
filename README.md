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

## `new` 分支：可靠性约束的目标域适配

本轮修改入口为 **`train.py`**；`train_v3.py` 保留为旧版本，不包含本轮修复。
保持网络参数名称和维度，可加载同一 backbone 的已有源域 checkpoint。
加载 checkpoint 后直接开始目标适配，不重新训练源模型，也不额外下载 ImageNet 权重。

```bash
python -u train.py --backbone resnet50 \
  --checkpoint /path/to/source_checkpoint.pth \
  --model_dir ./models/cast_resnet50_v4
```

请将 checkpoint 替换为实际源模型路径。数据目录仍使用原默认值，可用
`--source_root` 和 `--target_root` 指定。省略 checkpoint 会先训练源模型。
本次改动没有运行测试、训练或准确率评估；65% 是目标，不是已获得的结果。

本轮修复与改进：

- 修正类别概率排名方向，最高概率现在为 rank 1；恢复候选默认限制在教师前 2 类。
- 默认采用两视图最低置信度的分类别分位数阈值，并跨轮平滑；保留 `--threshold_mode cast`
  用于对照旧阈值策略（该选项并不复原全部旧代码）。
- 用固定原图/水平翻转对扫描目标训练集，减少随机视图造成的时序筛选波动。
- 恢复类别需源分类器或相对原型间隔支持；改写教师第一预测时两者都须支持。
  恢复标签暂不参与 DDRL，只有后续成为可信标签后才有资格参与特征对齐。
- 目标损失按选中数量归一化，保留置信度和恢复折扣的绝对作用；小批次支持下限为 8。
- 分两阶段校准骨干与分类头 BN，目标训练冻结全部 BN 统计量；只使用目标训练图像校准。
- 使用较温和的人脸增强、源域骨干较小学习率、梯度裁剪；MMD 距离计算移除三维大临时张量。

默认输出到新的 `cast_resnet50_v4` 目录，并记录 `config.json`。
`--augmentation legacy` 可单独恢复旧增强，`--bn_adapt_batches 0` 可关闭 BN 校准，
`--recovery_topk_per_class 0` 可关闭类别恢复，用于之后的消融实验。

评估沿用原分支流程：每轮在 `test/` 中选择最佳 checkpoint。因此 `*_best.pth` 对应的是
该集合上选模后的验证分数；`*_final.pth` 保存预设轮数的最终模型。
正式论文需说明此协议，不能把选模分数称为独立测试分数。

对照来源：[CAST 论文页面](https://ieeexplore.ieee.org/document/10843182)、
[论文对应公开实现](https://github.com/smwanghhh/CAST)。本轮核对了公开实现；
论文全文未能读取，不能据此声称完成逐节原文核验或证明方法首创。

