# 代码框架复核

代码版本：[`c67ac747`](https://github.com/tiantongtong1-oss/CAST/commit/c67ac74724a6b2083fb3901e599dc7bb6251201b)。代码提交后重新通过 GitHub 读取 `Networks.py`、`ema_utils.py`、`train.py`、`dataset.py`，再绘制本图。前三个已修改文件与本地通过测试的内容逐字一致，数据集文件沿用原分支。

![MobileNetV2 Dual-View EMA CAST](cast_mobilenetv2_framework.png)

下载：[PDF 矢量图](cast_mobilenetv2_framework.pdf) · [SVG 矢量图](cast_mobilenetv2_framework.svg)

## 图与代码对应

| 图中部分 | 实际代码 | 张量或作用 |
| --- | --- | --- |
| 源样本 | `dataset.RafDataSet` | RAF-DB 弱增强图像与真实标签 |
| 目标样本 | `dataset.FER.__getitem__` | 独立弱视图 1、弱视图 2、强视图；目标训练真标签不参与监督 |
| 1. Shared Student Backbone | `Networks.Model.feature` | MobileNetV2 → 全局平均池化 → Flatten → Dropout → Linear(1280,512) → Dropout |
| 2. EMA teacher | `ema_utils.create_ema_teacher/update_ema_teacher` | 整个学生网络的副本；参数及浮点 BN buffer 做 EMA，整型 buffer 复制；不接收梯度 |
| 全局统计 | `train.calculate_teacher_statistics` | 每个目标 epoch 扫描完整目标训练集，估计预测先验和逐类阈值 |
| 双视图筛选 | `train.generate_dual_view_pseudo_labels` + `ema_utils.select_dual_view_pseudo_labels` | 温度 softmax、逐视图分布校正、预测一致性、置信度阈值、严格回退、有限权重 |
| 3. DDRL | `Networks.Model.forward(task='target')` | 512 维学生特征的类条件域对齐 MK-MMD + 类间负 MK-MMD；可靠 mask、类别密度权重、有效类归一化 |
| 4. Classifier | `Networks.Model.fc/bn` | Linear(512,7,bias=False) → BatchNorm1d(7)；源/目标共用 |
| 4. Prototype memory | `ema_utils.PrototypeMemory.update` | 7 × 512；源教师特征先锚定，可靠目标双弱视图教师特征在学生 step 后更新 |
| 4. Target affinity | `ema_utils.PrototypeMemory.loss` | 学生强视图特征靠近本类原型，并通过相对余弦 margin 远离其他已初始化原型 |
| Classification loss | `train.run_training` + `weighted_mean_loss` | 源 CE 均值与目标加权 CE 分别归一化，目标权重逐步增加 |
| Classifier modulation | `train.classifier_modulation_loss` | 原 CAST 分类器权重余弦调制正则；完整损失中保留 `w3` |
| 反向/EMA | `train.backward_and_step` 后 `update_ema_teacher` | 先检查有限性、梯度裁剪和学生 step，再更新目标记忆与教师；EMA 不反传 |

图中的 DDRL 与 CCDR 是作用在同一组学生特征上的约束，不串接额外的特征变换层。教师直接读取增强图像，不读取学生 backbone 的输出。图中各颜色只区分类别，没有使用训练结果或准确率数据。

七类编号：`0 surprise, 1 fear, 2 disgust, 3 happy, 4 sad, 5 angry, 6 neutral`。

## 训练与推理顺序

1. 源预训练：源分类 CE + 源类间分离约束 + 分类器调制；选择源阶段最佳模型。
2. 初始化目标阶段：加载源权重，重新建立优化器/调度器，创建教师和空原型记忆。
3. 每个目标 epoch：全目标训练集双视图统计 → 阈值与预测先验。
4. 每个 batch：教师弱视图生成伪标签；学生对源弱视图和目标强视图前向；教师源特征更新源原型；计算完整损失并更新学生；可靠目标教师特征更新记忆；EMA 更新教师。
5. 验证集选择目标阶段最佳学生，测试集用于最终评估。推理只需学生 backbone 和分类器。

## 重新绘制

`architecture_manifest.json` 记录已复核代码的 SHA-256。绘图脚本先校验源码指纹，代码变化后需重新审查框架并更新指纹，避免继续输出过时图示。

```bash
python docs/draw_architecture.py
```

绘图依赖 `reportlab` 和 `PyMuPDF`，生成 PDF、SVG 和 3000 × 1775 PNG。图中文字和模块布局在 `draw_architecture.py` 中维护。
