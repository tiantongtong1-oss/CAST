"""计算全局双视图置信度阈值，并提供固定/动态 λ 和阈值 EMA 平滑。"""
import math


def lambda_for_epoch(epoch, ramp_epochs, start=0.9, end=1.0, fixed=None):
    """
    返回当前 epoch 使用的 λ。

    参数：
        epoch：目标域训练轮次，从 0 开始。
        ramp_epochs：λ 从 start 线性增长到 end 所需的轮数。
        start：动态 λ 的初始值。
        end：动态 λ 的最终值。
        fixed：固定 λ；不为 None 时覆盖动态调度。

    注意：
        这里只确定乘数 λ，不是最终置信度阈值 τ。
        start/end 是 λ 的调度端点，不是 τ 的上下限。
    """
    # 检查动态 λ 的两个端点：必须是有限正数。
    for value in (start, end):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('lambda endpoints must be finite and positive')

    # 只允许非递减调度；轮数至少为 1，epoch 不能为负数。
    if end < start or ramp_epochs < 1 or epoch < 0:
        raise ValueError('require end >= start, ramp_epochs >= 1 and epoch >= 0')

    # 如果指定固定 λ，例如 --threshold_lambda 0.9，
    # 每轮直接返回 0.9，不再使用动态增长公式。
    if fixed is not None:
        if not math.isfinite(fixed) or fixed <= 0:
            raise ValueError('fixed lambda must be finite and positive')
        return float(fixed)

    # 计算调度进度，最大为 1。
    # 例如 ramp_epochs=30：epoch=0 时进度为 0，epoch=29 时为 1。
    # 超过调度轮数后，进度保持为 1，λ 保持为 end。
    progress = min(epoch / max(ramp_epochs - 1, 1), 1.0)

    # 如果调度只有一轮，则直接使用最终值 end。
    if ramp_epochs == 1:
        progress = 1.0

    # 线性插值：λ = start + (end - start) × progress。
    return float(start + (end - start) * progress)


class GlobalConfidenceMean:
    """
    按样本数计算全部目标训练样本的全局置信度均值。

    调用方 train.py 为每个样本计算：
        score = min(原图最大预测概率, 弱增强最大预测概率)

    然后传入每个 batch 的 score 总和和样本数量。
    统计包含双视图预测类别不一致的样本，不先做一致性筛选。
    """

    def __init__(self):
        # 所有已统计样本的 score 总和。
        self.total = 0.0

        # 所有已统计的样本数量。
        self.count = 0

    def add_batch(self, confidence_sum, sample_count):
        """累计一个 batch 的置信度总和与样本数量。"""
        # 总和必须有限，样本数量不能为负数。
        if not math.isfinite(confidence_sum) or sample_count < 0:
            raise ValueError('invalid confidence statistics')

        self.total += float(confidence_sum)
        self.count += int(sample_count)

    def threshold(self, scale):
        """
        计算当前轮的原始阈值，scale 就是当前 λ。

        返回：
            raw_threshold：λ × 全局置信度均值。
            mean：全局置信度均值。
        """
        # 没有样本时无法计算均值。
        if self.count == 0:
            raise ValueError('target threshold loader is empty')

        # 总和除以总样本数，保证不同大小的 batch 按样本数加权。
        # 不是先求各 batch 均值再直接平均。
        mean = self.total / self.count

        # 原始阈值 τ_raw = λ × mean。
        # 这里没有对阈值设置固定上下限，也没有执行 clamp。
        return float(scale) * mean, mean


def smooth_global_threshold(raw_threshold, previous=None, momentum=0.8):
    """
    对不同 epoch 的全局阈值做 EMA 平滑。

    第一轮：
        τ[0] = raw_threshold[0]

    后续轮：
        τ[e] = momentum × τ[e-1]
             + (1 - momentum) × raw_threshold[e]

    previous 必须传入上一轮“平滑后的阈值”，而非上一轮原始阈值。
    返回值是当前轮实际使用的平滑阈值。
    """
    # 当前原始阈值必须是有限非负数。
    if not math.isfinite(raw_threshold) or raw_threshold < 0:
        raise ValueError('raw threshold must be finite and nonnegative')

    # EMA 动量必须位于 [0, 1)。
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError('momentum must be in [0,1)')

    # 第一轮没有历史阈值，直接使用当前原始阈值初始化。
    if previous is None:
        return float(raw_threshold)

    # 检查上一轮平滑阈值。
    if not math.isfinite(previous) or previous < 0:
        raise ValueError('previous threshold must be finite and nonnegative')

    # 默认 momentum=0.8：
    # 当前阈值 = 上一轮平滑阈值 × 0.8 + 当前原始阈值 × 0.2。
    # 此处同样没有设置固定上下限或裁剪。

    threshold = float(
        momentum * previous + (1.0 - momentum) * raw_threshold
    )

    print(f'当前双视图共用阈值是：{threshold:.6f}')
    return threshold
