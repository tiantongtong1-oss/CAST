# 类别平衡的覆盖率减侵入率公共半径

基于 v29 分支 experiment/bidirectional-balanced-common-radius-v29-run-history，
提交 96ea7f1f10ccb0687643e06f1b818ff367473b57。本文取代v29的半径分位数说明。

每次源域刷新，使用当前EMA Teacher提取L2归一化特征，保留各类均值原型。
对每个源域样本，d+为到真实类别原型的欧氏距离；d-为到最近其他有效类别
原型的欧氏距离。分别计算每类 P(d+ < R)、P(d- < R)，再在有效类别间等权平均。
选择覆盖率减侵入率最大的R，并填入全部class_radii。

算法遍历全部经验距离变化点，不使用任意粗网格。相同距离事件一起处理；
选择float32距离之后的最小可表示数，保证与严格 '<' 判定一致。
得分相同（1e-12数值容差）选最小半径；R=0也参与搜索，无正收益则拒绝全部
几何接纳。没有人为半径下限或覆盖率下限，没有额外权重超参数。
缺失类别不参与平均，也不作为竞争原型；其判断仍因未初始化而拒绝。
只有一个有效类别时其他类距离为正无穷；全部为空时R=0。

source_radius_quantile仅为旧命令兼容保留，不参与新半径选择，运行时无需传入。
source_refresh_interval仍控制刷新间隔。没有使用目标真实标签。
投票、竞争检查、稳定晋升、CE损失和运行归档继续沿用v29/v28。
特别是稳定晋升仍要求通过半径；此实验只改变公共半径选择。

日志新增 Radius_Mode: balanced_coverage_minus_intrusion，Common_Radius、
Class_Coverage_Strict、Balanced_Coverage_Strict、Class_Intrusion_Strict、
Balanced_Intrusion_Strict、Radius_Objective。侵入率是源域几何指标，
不是目标域伪标签错误率。class_radii继续随目标最佳检查点保存；诊断指标写日志，
也保存到检查点的radius_selection字段，便于离线检查。
旧检查点 --checkpoint 仍只用于初始化模型权重，不是完整恢复训练。

## RAF → FER（使用现有RAF源域模型）

```bash
python -u train.py \
  --data1 rafdb --data2 fer \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 \
  --checkpoint /workspace/ttt/code/CAST/model_history/experiment__global-mean-std-cast-ce-v27-run-history__20261028-094557-500245__55f189/models/mobilenet_v2_rafdb_fer_global_mean_std_cast_ce_v27_pseudo_metrics_source_best.pth \
  --pre_epochs 0 --epochs 30 --lr 0.0001 --workers 0 \
  --source_refresh_interval 1 \
  --threshold_std_scale 0.5 --threshold_smoothing 0.8 \
  --threshold_min 0.85 --threshold_max 0.99 \
  --run_name raf_to_fer_discriminative_radius_v30
```

FER → RAF同样支持：交换data1/data2与数据路径；从头预训练使用pre_epochs=30，
不传checkpoint。不能把上述RAF源域模型称为FER源域预训练模型。

## 验证

新增测试覆盖：可分离分布、相同距离、无正收益、类别样本复制不变性、
随机分布与逐候选暴力搜索对比、缺失类别、输入校验。
运行 python -m unittest discover -s tests -p test_discriminative_radius.py -v。
当前环境缺少PyTorch；半径库真实集成测试跳过，完整GPU训练效果未验证。
