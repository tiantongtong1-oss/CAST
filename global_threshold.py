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
