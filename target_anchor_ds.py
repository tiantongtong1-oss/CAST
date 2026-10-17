"""Current-view validation of historical target anchor senders; no target truth."""
import numpy as np


def anchor_reference_scores(labels, confidence, region_ok, history_labels,
                            history_quality, thresholds):
    labels = np.asarray(labels, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float64)
    region_ok = np.asarray(region_ok, dtype=bool)
    history_labels = np.asarray(history_labels, dtype=np.int64)
    history_quality = np.asarray(history_quality, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if labels.ndim != 1 or any(a.shape != labels.shape for a in
                              (confidence, region_ok, history_labels, history_quality)):
        raise ValueError('anchor observations must have matching vector shapes')
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all():
        raise ValueError('thresholds must be a finite class vector')
    safe = labels.clip(0, len(thresholds)-1)
    keep = (labels >= 0) & (labels < len(thresholds)) & (labels == history_labels)
    keep &= region_ok & np.isfinite(confidence) & (confidence >= thresholds[safe]) & (confidence <= 1)
    keep &= np.isfinite(history_quality) & (history_quality > 0) & (history_quality <= 1)
    return np.where(keep, np.minimum(confidence, history_quality), 0.)
