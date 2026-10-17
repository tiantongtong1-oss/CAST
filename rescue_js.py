"""Offline-compatible probability consistency gate for rescue candidates only."""
import numpy as np
from ds_diagnostics import view_probability_metrics


def rescue_js_gate(probabilities1, probabilities2, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= np.log(2.):
        raise ValueError('rescue JS threshold must be finite and in [0, ln(2)]')
    js, _ = view_probability_metrics(probabilities1, probabilities2)
    return js, js <= threshold



def rescue_js_enabled(threshold, epoch, start_epoch=0):
    """Schedule only; omitted threshold disables JS regardless of start epoch."""
    if start_epoch < 0 or epoch < 0:
        raise ValueError('epoch and start_epoch must be nonnegative')
    return threshold is not None and epoch >= start_epoch
