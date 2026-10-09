# v8-02：逐类独立方差

基于 experiment/role-separated-sparse-knn-v8-01，提交 3026cbbc74a6b83798248decd8278cf99f89f9e4。
目标分支：experiment/role-separated-sparse-knn-v8-02。

## 方差定义

保留各向同性高斯，但每个类别独立估计一个标量方差，而不是所有类别共用一个标量，也不是逐维对角协方差。
对 L2 归一化后的源域特征 z，按源域真实标签分组：

```text
mu_c = mean(z_i | y_i=c)
observed_var_c = max(mean_d(mean_i(z_i[d]^2) - mu_c[d]^2), variance_floor)
var_c = beta * previous_var_c + (1-beta) * observed_var_c
```

首次有效刷新直接初始化。每类至少需要两个有效源域样本；不足时该类本轮不可用于高斯判断，恢复后重新初始化。各类共用原有 EMA 动量日程，但方差值独立更新。源域均值仍每次直接重估。目标域标签和特征不用于方差估计。

- 高斯归属：||z-mu_c||² / var_c，与原 chi-square 阈值比较。
- 默认 support 模式：每个邻居按其自身类别 c 使用 vector_var_c = D * var_c，核带宽平方 = multiplier² * vector_var_c，距离上限 = 2 * radius_multiplier * vector_var_c。不同类别的竞争投票各用自己的尺度；软标签判断与同一套投票一致。
- v7 评分模式：候选样本及其邻居对候选类别进行高斯判断，核带宽使用候选类别 var_c。
- class_variances 每行重复对应类别的标量方差；日志增加 Class_Variances、Observed_Class_Variances、Classes_Ready。
- global_var、sigma 等历史名字保留为汇总诊断及 API 兼容项，不参与门控、核带宽或距离上限计算。旧实验说明中的全局方差描述仅适用于旧分支。
- 保留 v8-01 的 --immediate_knn_prototype_update 开关及原有默认值、资格分离、损失、模型选择和日志频率。

## 运行

在已有 CAST 仓库根目录执行：

```bash
git fetch origin
git switch --track origin/experiment/role-separated-sparse-knn-v8-02
python -u train.py \
  --checkpoint /path/to/source_best.pth \
  --source_path /path/to/raf-basic \
  --target_path /path/to/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 0 --epochs 30 \
  --immediate_knn_prototype_update \
  --run_name role_sparse_v8_02_class_variance
```

路径换成实际位置，backbone 必须匹配 checkpoint。对比 v8-01 时保持源域 checkpoint、数据划分、随机种子、训练预算及其他参数相同；如果 v8-01 没开启即时原型更新，这里也删除该开关。--checkpoint 仅初始化模型权重，不是恢复完整训练；高斯库会重新由源数据建立。

```bash
python -m unittest discover -s tests -v
```

测试覆盖独立方差估计、逐类 EMA、类别缺失与重新初始化、按类别归属判断、发送者类别核尺度/距离上限，以及原有角色与即时原型更新训练接线。使用合成数据验证，不代表真实 RAF→FER 性能提升。

验证结果：49 项测试全部通过（CPU 合成数据）；同时修正原测试仍要求完整 mu 向量输出的旧断言，使之符合 v8-01 已采用的紧凑日志。
