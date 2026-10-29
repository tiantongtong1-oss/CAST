# 类别平衡公共半径

基于 experiment/bidirectional-cast-ce-v28-run-history，提交
f2af25c8554f61b8ee965efd5f63608c6f28701c。

每次源域刷新仍由当前 EMA Teacher 提取归一化特征，并保留各类自己的均值原型。
汇总每个源域样本到真实类别原型的欧氏距离；每个有效类别的总权重相同，
类内每个样本的权重为 1 / (有效类别数 × 本类有效样本数)。按距离排序后，
取累积权重首次达到 source_radius_quantile 的观测距离作为公共半径 R。
默认 q=0.90。所有 class_radii 元素均保存 R，不再分别计算七个分位数。

这相当于使各类源域覆盖率的平均值达到约90%，不保证每类分别覆盖90%。
新分位数是经验CDF的逆，不使用旧实现的线性插值。目标判断保留 distance < R，
所以边界同值样本不通过，实际严格覆盖率可能低于 q。日志会记录实际严格覆盖率。
缺失类别不参与权重，仍标记未初始化并拒绝其几何判断；全空刷新半径为0。

train.log 新增 Radius_Mode、Common_Radius、Class_Coverage_Strict、
Balanced_Coverage_Strict。现有 Class_Radii 输出及目标最佳检查点
reliability_bank.class_radii 均保留，七个值相同。覆盖率仅用于本轮日志。
双向数据加载、置信度、投票、竞争检查、晋升、损失及运行归档继续使用v28逻辑。
输入 --checkpoint 只初始化模型权重，不恢复完整训练状态。

## 运行：RAF-DB → FER2013

```bash
python -u train.py \
  --data1 rafdb --data2 fer \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 --pre_epochs 30 --epochs 30 \
  --lr 0.0001 --workers 0 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --threshold_std_scale 0.5 --threshold_smoothing 0.8 \
  --threshold_min 0.85 --threshold_max 0.99 \
  --run_name raf_to_fer_balanced_radius_v29
```

## 运行：FER2013 → RAF-DB

```bash
python -u train.py \
  --data1 fer --data2 rafdb \
  --source_path /workspace/ttt/code/data/fer2013 \
  --target_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --backbone mobilenet_v2 --pre_epochs 30 --epochs 30 \
  --lr 0.0001 --workers 0 \
  --raf_val_fraction 0.10 --raf_split_seed 2000 \
  --source_radius_quantile 0.90 --source_refresh_interval 1 \
  --threshold_std_scale 0.5 --threshold_smoothing 0.8 \
  --threshold_min 0.85 --threshold_max 0.99 \
  --run_name fer_to_raf_balanced_radius_v29
```

两条命令显式选择 lr=0.0001，与v28文档示例一致；程序默认学习率仍为0.001。
若已有匹配骨干及源域的检查点，增加 --checkpoint /实际路径/source_best.pth，
并将 --pre_epochs 30 改为 --pre_epochs 0。RAF作目标时继续从官方训练集划分
验证集，官方测试集仅用于最终评估。归档默认保存到 ./model_history。

## 验证

```bash
python -m unittest discover -s tests -p test_balanced_radius.py -v
python -m unittest discover -s tests -p test_bidirectional_data.py -v
python -m unittest discover -s tests -p test_global_mean_std_ema.py -v
python -m unittest discover -s tests -p test_cast_classification.py -v
python -m unittest discover -s tests -p test_run_history.py -v
python -m py_compile balanced_radius.py source_prototype_radius.py train.py
```

开发环境没有PyTorch，算法测试可执行，真实半径库刷新/保存测试自动跳过。
完整GPU训练和实际数据效果需在训练机器验证。
