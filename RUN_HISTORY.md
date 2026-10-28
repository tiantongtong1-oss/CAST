# 每次训练自动归档

基于 experiment/global-mean-std-cast-ce-v27，基线提交
37f0d8e55f56ce41de68f0884fa0d04ee4db4996。
本分支仅增加运行归档，不包含其他分支的伪标签统计或离线测评功能。
阈值、CE、补救筛选、EMA、原型和模型选择规则保持 v27 行为。

每次调用 train.py（参数解析成功后）自动创建：

```text
model_history/
  experiment__global-mean-std-cast-ce-v27-run-history__时间戳__唯一后缀/
    logs/
      train.log
      <run_name>_ds_diagnostics.csv
    models/
      <backbone>_rafdb_fer_<run_name>_source_best.pth
      <backbone>_rafdb_fer_<run_name>_target_best.pth
    BRANCH_CHANGES.md
    branch_changes.patch
    working_tree.patch
    run.json
```

目录使用实际 Git 分支名，斜杠转换为双下划线。时间使用 UTC+8，精确到微秒，
并附随机后缀以隔离同时运行的进程。每次运行都创建新目录，不覆盖以前结果。
没有 Git 时使用 unknown-branch；分离 HEAD 时使用 detached-<commit>。

- logs/train.log 自动记录训练入口中的 Python stdout/stderr，并在控制台同步显示；
  不再需要外部 tee。参数解析或模块导入时的错误发生在归档入口之前，不在此日志中。
  原生库直接向文件描述符写出的内容不保证由 Python 日志捕获。
- models 保存本次生成的最佳检查点；pre_epochs=0 时复制已有源域检查点到本次目录，
  原输入文件保留，目标阶段初始化和优化器规则不变。复制会额外占用一个检查点大小的磁盘空间。
- BRANCH_CHANGES.md 自动记录当前分支、提交、相对固定 v27 基线的提交说明和文件变化统计。
- branch_changes.patch 保存基线到 HEAD 的已提交差异；working_tree.patch 保存 HEAD
  到当前已跟踪文件的未提交差异（含暂存区变化）。未跟踪文件只记录文件名，不保存内容。
- run.json 保存参数、命令、起止时间、版本、输入检查点、归档目录和运行状态。
  成功为 completed，异常为 failed，Ctrl+C 为 interrupted；异常堆栈写入日志。
  强制终止、断电等无法执行清理时，状态会保留为 running。
- Git 或基线不可用时，报告明确标记不可用；训练仍可运行。
- 默认相对当前工作目录保存到 ./model_history；可用 --history_root 更换父目录。
  默认目录已加入 .gitignore；自定义目录若放在仓库内，应自行加入忽略规则。

运行示例（保留原 v27 参数，无需 tee）：

```bash
python -u train.py \
  --backbone mobilenet_v2 \
  --source_path /workspace/ttt/code/test-upload-clean/datesets/raf-basic \
  --target_path /workspace/ttt/code/data/fer2013 \
  --checkpoint ./models/rafdb_fer/mobilenet_v2_rafdb_fer_source_uniform_control_v27_source_best.pth \
  --pre_epochs 0 --epochs 30 --ema_decay 0.999 \
  --threshold_std_scale 0.5 --threshold_smoothing 0.8 \
  --threshold_min 0.85 --threshold_max 0.99 \
  --run_name global_mean_std_cast_ce_v27
```

模型仍是最佳检查点策略，不会额外保存每一轮。原有 ./models 中的检查点仍可作为输入。
后续要加载本次结果，请使用控制台 Run_History 输出目录下 models 中的实际路径。

验证：
```bash
python -m unittest discover -s tests -p test_run_history.py -v
python -m unittest discover -s tests -p test_cast_classification.py -v
python -m unittest discover -s tests -p test_global_mean_std_ema.py -v
python -m py_compile train.py run_history.py
```
归档测试包含真实临时 Git 仓库、分支名、代码差异、连续运行隔离、日志双写、
异常与中断状态，以及训练入口接线。完整 GPU/数据训练未在开发环境运行。
