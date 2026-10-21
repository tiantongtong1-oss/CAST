# 校准隔离与支持度阈值消融

基于 v18 的 5ab772b2966d659bd851f4e183ed360853f88327。
新分支 experiment/mean-support-calibration-isolation-v19。

校准使用独立 FER 实例、双弱视图、固定顺序、num_workers=0、独立 generator。
不生成无用的 strong view；不读取目标标签参与筛选。
Python/NumPy/PyTorch CPU 和可用 CUDA RNG 在校准后恢复（异常也恢复）。
训练 target loader 使用独立 generator。三种模式均扫描校准集，便于公平比较。
这不保证与历史 v18 运行逐 batch 相同，也不消除轮内 Teacher 更新导致的分布漂移。

--support_threshold_mode otsu：A，默认，保留原始 Otsu 行为。
--support_threshold_mode fixed：B，固定阈值。
--support_threshold_mode otsu_guarded：C，不可分时回退固定阈值。

固定阈值 --support_fixed_threshold 默认 .92。
C 的实验性默认门槛：--support_min_count 128，--support_min_spread .01
（P95-P05），--support_min_separation .7（类间方差/总方差），
--support_min_group_fraction .05。门槛不是验证过的最优参数。
空、常数、样本少、分布窄、分离弱或小组占比不足时使用固定阈值。
回退仍需 available 且 support>=固定阈值；常数低分不会全部放行。

日志输出分位数、分离度、最小组占比、原始 Otsu、实际阈值、模式、回退原因、补救通过率。
原置信度、半径、损失、角色分离和原型写入开关不变。
与既有日志比较应保留 --immediate_prototype_update；原型消融另做。
每种模式使用不同 run_name；从同一 source checkpoint 开始，不是恢复旧 target 训练。

验证：7 项 NumPy 测试通过；2 项真实 PyTorch 随机隔离测试因环境缺少 torch 跳过。
语法检查通过。未运行 RAF/FER 或 CUDA 训练。
有 torch 的训练环境运行：

```bash
python -m unittest discover -s tests -p 'test_calibration_isolation.py' -v
python -m unittest discover -s tests -p 'test_neighbor_support.py' -v
```

先比较 A/B，再评估 C。使用多种子和验证集选择规则，最终测试集仅评估。
