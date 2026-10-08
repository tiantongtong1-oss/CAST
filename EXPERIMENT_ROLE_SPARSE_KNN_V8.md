# v8：资格分离、稀疏可靠支持与可选邻域软标签

## 基线与范围

- 基于 `experiment/source-global-gaussian-knn-reliability-v7`，提交 `a56bfcb3cbf600a52b283e424406cf0d40905b48`。
- 新分支：`experiment/role-separated-sparse-knn-v8`。
- 任务仍为 RAF-DB → FER2013、7 类静态表情分类；骨干网络接口保持兼容。
- 默认启用资格分离和新的 support 评分；邻域软标签默认关闭，用于第三阶段消融。
- 本次验证包括单元测试与合成数据 CPU 训练集成测试，未完成真实 FER 数据训练，不宣称性能提升。

## 相对于 v7 的变化

| 位置 | v7 | v8 |
| --- | --- | --- |
| 分类/原型更新资格 | `final_mask` 同时控制学习、MMD、原型更新 | `learn_mask`（代码中仍叫 `final_mask/con_idx`）用于学习，`anchor_mask` 用于 MMD 和更新原型 |
| 邻域发送者 | 当前 Teacher 高置信度且属于源域高斯区域 | 默认要求连续稳定资格、当前记忆标签一致、属于源域高斯区域 |
| 稀疏救回 | score 乘以 density 和 sparse_penalty | 密度仅作诊断；由可靠支持数、有效支持数、纯度、优势和距离限制判断 |
| 核带宽 | 每维 variance 直接作用于完整向量距离 | `h² = bandwidth_multiplier² * D * global_var`；与高斯区域阈值分开 |
| 邻域输出 | 只确认 Teacher 原标签 | 输出所有类别的可靠邻域分布，可选软监督冲突样本 |
| 日志 | 总接纳、救回、原型计数 | 新增发送者、晋升锚点、稀疏救回、软修正数以及 UAR/Macro-F1 |
| 运行 | CUDA、固定 checkpoint 后缀 | 默认 CUDA；支持 CPU smoke test 与 `--run_name` 隔离消融 |

新增 `reliability_roles.py`、`sparse_reliable_knn.py` 和两组测试，主要接线位于 `train.py`。`Networks.py`、`prototype_utils.py` 的接口无需修改。

### 1. 资格分离与晋升

训练中的高置信度接纳与 KNN OR rescue 保留；双弱视图不一致时仍不走硬标签救回。

| 资格 | 控制哪些操作 | 默认条件 |
| --- | --- | --- |
| 学习资格 | 硬标签分类损失、类平衡原型一致性损失 | 通过双视图置信度，或通过 KNN rescue |
| 锚点资格 | `PrototypeBank.update_target()`、同类 MMD 对齐和类间分离中的目标样本 | 连续 3 个 epoch 被接纳、预测类别不变，且位于相应源域高斯区域 |
| 发送资格 | 下一次 memory refresh 后的 KNN 投票 | 上一个 epoch 已晋升、当前记忆视图预测类别未变、仍在源域高斯区域 |

刚救回的样本可以向可靠原型学习，但不会立刻改变原型。`anchor_mask` 是二值量，符合 `Networks.forward()` 的 `idx == 1` 约定；发送分数单独存储，不将浮点可靠性硬塞进二值接口。

历史以 target-train 稳定 sample ID 索引，保存在 `ReliabilityRoles` 的 labels/streak/last_epoch/quality buffers 中：

- 同一 epoch 重复看到同一个样本不会累计晋升次数。
- 标签改变、未被接纳、离开高斯区域、跨 epoch 缺失会重置或撤销资格。
- 发送分数取「上一训练观测的双弱视图最小置信度」与「当前确定性 memory 视图置信度」的较小值。它是可靠性启发式，不是经校准的正确率。
- 邻域发送资格在 refresh 时冻结，防止新救回样本在同一次 refresh 内立刻成为支持来源。默认每个 epoch refresh；增大间隔会延迟发送者更新/撤销。
- 所有高置信度样本也必须通过晋升和区域条件才能更新公共原型；但其分类学习仍保留 v7 的 OR 通道行为。

默认 `promotion_epochs=3`、`knn_warmup_epochs=3`：高置信度样本可在 epoch 0/1/2 积累资格，epoch 3 的刷新开始提供稳定邻域支持。早期若没有可靠发送者，救回关闭是预期行为，不会自动降级为不可靠投票。

### 2. 为什么 v7 的稀疏样本过不了，v8 怎样处理

v7 默认 `density < 0.5` 时：

```text
score <= density * sparse_penalty < 0.5 * 0.5 = 0.25
score_threshold = 0.5
```

所以所有稀疏候选都无法通过 rescue。v8 的 `support` 模式不再把绝对密度乘入 score。

对完整 target memory 先选最多 k 个最近邻（排除相同 sample ID，不按类别预过滤）。设：

```text
d²_ij = 2 - 2*cos(z_i,z_j)
h² = bandwidth_multiplier² * D * global_var
w_ij = exp(-d²_ij / (2*h²))
radius² = 2 * radius_multiplier * D * global_var
```

只有具备发送资格、处于自身预测类别源域高斯区域、且 `d²_ij <= radius²` 的邻居可以投票。不同类别的可靠发送者全部进入投票分母，防止只检索同类产生虚假的纯度。

```text
v_ij = w_ij * sender_score_j * valid_sender_ij
p_i(c) = sum_j(v_ij * [label_j=c]) / sum_j(v_ij)
support_count_c = 同类有效发送者数
effective_support_c = sum_j(v_ij*[label_j=c])²
                      / sum_j((v_ij*[label_j=c])²)
quality_c = sum_j(w_ij*sender_score_j*[valid sender of c])
            / sum_j(w_ij*[valid sender of c])
score_c = query_in_source_region_c * p_i(c) * quality_c
```

分母为零时安全拒绝；代码中使用 epsilon 保持数值稳定。默认通过条件为：

- 至少 3 个同类可靠邻居；
- 有效支持数至少 2，避免一个邻居独占权重；
- 同类投票纯度至少 0.8；
- 最高类与次高类的投票差至少 0.2；
- 最高类与 Teacher 硬伪标签一致；
- query 位于该类源域高斯区域；
- score 至少 0.5。

`density_threshold` 现在只区分日志中的 dense/sparse。`knn_sparse_penalty` 仅在 `--knn_score_mode v7` 中生效。稀疏一致样本有可能通过，密集冲突样本也会被拒绝；不是所有稀疏样本都被接纳。

共享 source variance、source class mean 和名义 chi-square 区域仍沿用 v7。`D*variance` 是可解释的尺度修正，不是跨域最优带宽的证明。仍需对带宽、半径和支持数做消融。

### 3. 邻域软标签：默认关闭

启用 `--neighbor_soft_labels` 后，仅为以下样本增加软监督：

- 双弱视图类别一致，但未通过任何硬标签接纳通道；
- 可靠邻域的获胜类别与 Teacher 不同；
- 邻域满足上述支持数、有效支持数、纯度、优势和 score 条件；
- query 在邻域获胜类别的源域高斯区域内；
- KNN warmup 已结束。

```text
q_soft = (1 - neighbor_soft_mix) * mean(teacher_weak1, teacher_weak2)
         + neighbor_soft_mix * neighbor_distribution
L_soft = mean(-sum_c q_soft[c] * log student_probability[c])
```

默认 mix=0.5、loss weight=0.1。软目标 detach，不向 Teacher 或 memory 反传。软修正样本不进入硬标签 CE、原型一致性损失、MMD、原型更新或本轮资格晋升。它在后续 epoch 获得稳定的 Teacher 硬标签证据后，才能走正常晋升流程。已有高置信度通道不会被软标签强制覆盖。

### 4. 其他修正、日志和协议

- 修复 `class_variances[ready].fill_(...)`：布尔高级索引产生副本，原写法未更新原 buffer。改为索引赋值。实际 gate 使用 `global_var`，因此这个修复主要影响诊断/保存的方差 buffer。
- 新增 `Anchor_Num`、`Anchor_Distribution`、`Stable_Senders`、`Sparse_Rescue_Num`、`Soft_Repair_Num`、`Soft_Loss`。
- 验证/测试输出逐类召回与支持数、UAR（有真值样本的类别均值）、Macro-F1（所有类别均值）。仍按目标 validation accuracy 选模型，未改变 v7 的选择标准。
- target-train 真标签仍不用于高斯、角色、邻域或损失；目标 validation 标签仍用于选择 checkpoint。因此不是“完全不使用目标标签选模型”的严格协议。
- 保留固定源原型与动态 Teacher 的 v7 设计；潜在特征空间漂移不属于本次解决范围。
- 模型参数和 BN 仍共享：资格分离限制显式标签/原型/MMD信息，不保证彻底隔离所有错误影响，也不保证所有长尾类别都能获救。
- `--checkpoint` 仍是模型权重初始化，不是完整 resume。新 checkpoint 保存 role_tracker，但当前入口不会恢复优化器、memory 或资格轨迹。只加载可信来源的 PyTorch checkpoint。

## 环境与测试

使用已有 v7 环境即可；核心依赖包括 PyTorch、torchvision、numpy、pandas、opencv-python、scikit-learn、matplotlib、Pillow。建议安装匹配的 torch/torchvision 版本。

```bash
python -m unittest discover -s tests -v
```

本次验证：Python 3.12、CPU PyTorch 2.14.1、torchvision 0.29.1，43 项测试通过。其中 19 项为 v8 新测试方法；训练集成测试实际执行 1 个 source epoch + 4 个 target epoch，分别覆盖默认配置和 soft-label 开关，使用小型合成数据与替代骨干，不代表真实 FER 性能。

覆盖：稀疏可信默认通过、v7 稀疏默认拒绝、密集冲突拒绝、低可靠发送者屏蔽、有效支持数、距离保护、自身排除、NaN/空库安全、分块与记忆顺序一致、晋升/撤销/历史保存、目标标签不参与刷新、软目标梯度隔离及完整训练接线。

## 获取分支

已有 CAST 仓库时，在仓库根目录运行：

```bash
git fetch origin
git switch --track origin/experiment/role-separated-sparse-knn-v8
```

如果本地已经有此分支，使用 `git switch experiment/role-separated-sparse-knn-v8`。新目录可使用：

```bash
git clone --branch experiment/role-separated-sparse-knn-v8 https://github.com/tiantongtong1-oss/CAST.git CAST-v8
cd CAST-v8
```

## 推荐主实验：资格分离 + 稀疏可靠支持

以下路径沿用 v7 的运行示例，请按实际数据位置调整。FER2013 需要独立的 train/val/test 子目录，类别文件夹映射由 dataset.py 处理。

```bash
mkdir -p new_logs
set -o pipefail
python -u train.py \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 \
  --pre_epochs 30 --epochs 30 \
  --promotion_epochs 3 \
  --knn_score_mode support \
  --knn_min_support 3 --knn_min_effective_support 2 \
  --knn_min_purity 0.8 --knn_min_margin 0.2 \
  --knn_score_threshold 0.5 --knn_radius_multiplier 2 \
  --knn_warmup_epochs 3 --knn_refresh_interval 1 \
  --run_name role_sparse_v8 \
  2>&1 | tee new_logs/role_sparse_v8.log
```

## 复用 v7 源域 checkpoint

只复用相同骨干、同样类别映射的 source checkpoint。下面的路径是 v7 默认保存名称；如实际文件名不同请替换。

```bash
python -u train.py \
  --checkpoint ./models/rafdb_fer/mobilenet_v2_rafdb_fer_source_global_gaussian_knn_v7_source_best.pth \
  --pre_epochs 0 --epochs 30 \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --backbone mobilenet_v2 \
  --run_name role_sparse_v8_from_source
```

## 分阶段消融

下列参数追加到相同数据路径、骨干和 source checkpoint 命令中。每次使用不同 run_name，避免覆盖模型。

| 实验 | 参数 |
| --- | --- |
| v7 算法基线 | `--no_role_separation --knn_score_mode v7 --run_name v7_baseline` |
| 仅分离学习/锚点资格 | `--knn_score_mode v7 --run_name role_only` |
| 仅新稀疏支持评分 | `--no_role_separation --knn_score_mode support --run_name sparse_only` |
| 完整 v8，推荐先跑 | `--knn_score_mode support --run_name role_sparse_v8` |
| 完整 v8 + 软标签 | `--knn_score_mode support --neighbor_soft_labels --neighbor_soft_mix 0.5 --neighbor_soft_weight 0.1 --run_name role_sparse_soft_v8` |
| 无 KNN 且无资格分离 | `--no_knn_gate --no_role_separation --run_name confidence_baseline` |

`v7` 评分模式原样保留其即时置信度邻居选择，不使用历史发送分数；“仅分离资格”消融只改变原型更新和 MMD 参与者。`support` 模式才启用发送分数；关闭资格分离时，该分数退化为当前高置信度且处于源高斯区域的邻居置信度。

优先比较每类 recall、UAR、Macro-F1、救回数量和原型资格数量。`Sparse_Rescue_Num > 0` 只说明通道工作，不证明被救回标签正确。需要离线诊断伪标签质量时，请把 target 真标签限制在独立评估代码中，不能反馈到训练筛选。
