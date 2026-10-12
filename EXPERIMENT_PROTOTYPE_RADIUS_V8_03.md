# v8-03：直接按预测类别原型距离筛选（无目标域邻居投票）

分支：`experiment/role-separated-sparse-knn-v8-03-prototype-radius`。
本次更正此前保留目标域 kNN 投票的实现。当前入口 `train.py` 仅使用
`source_prototype_radius.SourcePrototypeRadiusBank` 进行分布判断。

## 类别阈值

每次刷新，用当前 EMA Teacher 提取确定性视图的源域特征，按真实标签分组。
令 z_i 为 L2 归一化后的源域特征：

```text
mu_c = mean(z_i | y_i=c)
d_i = ||z_i - mu_c||_2
R_c = mean(largest min(11, N_c) distances d_i)
```

原型 mu_c 保持源域归一化特征的算术均值，不再单独归一化。
最远的 11 个源域样本只用来确定 R_c；没有目标域近邻查询、投票、纯度或密度评分。
少于 11 个有效源域样本时使用全部；缺失类别不能通过筛选。
零半径、非有限特征、零向量和越界标签均不能通过严格小于判断。
源域原型和阈值默认每个目标域 epoch 重新计算，不计算方差或方差 EMA。

## 伪标签接纳

Teacher 两个弱增强视图的预测类别相同，记为 c。保持原有双视图特征融合：

```text
z_t = normalize(normalize(feature_weak1) + normalize(feature_weak2))
d_t = ||z_t - mu_c||_2
region_ok = d_t < R_c
final_mask = two_view_agreement AND region_ok
```

例如预测为第六类，仅比较到第六类源域原型的距离与 R_6。
类别索引沿用数据集原有 0-based 映射。
高置信度不再通过 OR 分支绕过距离判断，低置信度也不需要邻居支持。
最终接纳样本参与目标域分类损失和原型一致性损失。

保留 `PrototypeBank` 的原型损失、目标域原型 EMA 和角色晋升机制。
几何筛选始终使用源域原型；目标域混合原型用于损失，不改变源域阈值。
默认连续 `--promotion_epochs 3` 轮通过后才成为共享知识更新的 anchor。
`--immediate_prototype_update` 允许所有距离接纳样本立即更新目标域原型，
不改变 MMD anchor 的晋升规则。

## 参数与日志

- `--source_radius_k 11`：源域每类最远距离的数量，默认 11。
- `--source_refresh_interval 1`：源域原型与阈值刷新间隔，默认每个 epoch。
- `--immediate_prototype_update`：即时更新目标域原型。
- 删除训练入口的 `--knn_k`、邻居评分、邻居软标签和 kNN warmup 参数。
- 原 `--immediate_knn_prototype_update` 改为 `--immediate_prototype_update`。
- 日志输出 `Class_Radii`、`Agreement_Num`、`Radius_Pass_Num`、`Final_Accept_Num`。
  `Radius_Pass_Num` 仅统计距离通过数，最终接纳还要求双视图一致。

旧 kNN/Gaussian 模块保留为历史代码，当前训练入口不导入、不调用。
旧测试中的部分训练入口和参数断言属于此前版本；本次按用户要求未运行或新增测试，
此前“55 项通过”的结果不适用于本次修改。

## 从源域预训练开始运行并保存日志

```bash
git fetch origin
git switch experiment/role-separated-sparse-knn-v8-03-prototype-radius
git pull --ff-only
mkdir -p logs
set -o pipefail
python -u train.py \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --data1 rafdb --data2 fer \
  --backbone mobilenet_v2 \
  --pre_epochs 30 --epochs 30 \
  --source_radius_k 11 \
  --immediate_prototype_update \
  --run_name source_radius11_direct \
  2>&1 | tee "logs/source_radius11_direct_$(date +%Y%m%d_%H%M%S).log"
```

不传 checkpoint：先做 30 轮源域预训练，再做 30 轮目标域训练。
本次未运行测试或真实数据训练，不报告准确率结果。
