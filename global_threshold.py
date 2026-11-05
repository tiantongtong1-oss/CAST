"""Global two-view confidence threshold and progress-dependent lambda."""
import math


def lambda_for_epoch(epoch, ramp_epochs, start=0.9, end=1.0, fixed=None):
    """Linear tightening schedule; no claim of optimal lambda estimation."""
    for value in (start, end):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('lambda endpoints must be finite and positive')
    if end < start or ramp_epochs < 1 or epoch < 0:
        raise ValueError('require end >= start, ramp_epochs >= 1 and epoch >= 0')
    if fixed is not None:
        if not math.isfinite(fixed) or fixed <= 0:
            raise ValueError('fixed lambda must be finite and positive')
        return float(fixed)
    progress = min(epoch / max(ramp_epochs - 1, 1), 1.0)
    if ramp_epochs == 1:
        progress = 1.0
    return float(start + (end - start) * progress)


class GlobalConfidenceMean:
    """Sample-weighted mean across batches, including disagreeing predictions."""
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add_batch(self, confidence_sum, sample_count):
        if not math.isfinite(confidence_sum) or sample_count < 0:
            raise ValueError('invalid confidence statistics')
        self.total += float(confidence_sum)
        self.count += int(sample_count)

    def threshold(self, scale):
        if self.count == 0:
            raise ValueError('target threshold loader is empty')
        mean = self.total / self.count
        return float(scale) * mean, mean



def smooth_global_threshold(raw_threshold, previous=None, momentum=0.8):
    """EMA of thresholds across epochs; first epoch uses its raw value directly."""
    if not math.isfinite(raw_threshold) or raw_threshold < 0:
        raise ValueError('raw threshold must be finite and nonnegative')
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError('momentum must be in [0,1)')
    if previous is None:
        return float(raw_threshold)
    if not math.isfinite(previous) or previous < 0:
        raise ValueError('previous threshold must be finite and nonnegative')
    return float(momentum * previous + (1.0 - momentum) * raw_threshold)
