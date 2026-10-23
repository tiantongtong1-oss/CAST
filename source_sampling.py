"""Mild source-only resampling probabilities from source training labels."""
import numpy as np


def mild_sampling_weights(labels, classes=7, power=0.5):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not len(labels) or not np.issubdtype(labels.dtype,np.integer):
        raise ValueError('labels must be a nonempty integer vector')
    if ((labels<0)|(labels>=classes)).any():
        raise ValueError('source label out of range')
    if not np.isfinite(power) or not 0<=power<=1:
        raise ValueError('sampling power must be finite and in [0,1]')
    counts=np.bincount(labels,minlength=classes)
    class_weights=np.zeros(classes,dtype=np.float64)
    present=counts>0
    class_weights[present]=counts[present].astype(np.float64)**(-power)
    weights=class_weights[labels]
    expected=len(labels)*counts*class_weights/weights.sum()
    return weights,counts,expected
