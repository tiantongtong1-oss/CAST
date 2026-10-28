"""Read-only target-training pseudo-label diagnostics (CAST class order)."""
import csv
import numpy as np


CLASS_NAMES = ('Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral')
STAGES = ('confidence', 'rescued', 'final', 'supervised', 'anchors')
COLUMNS = ('epoch', 'stage', 'class_id', 'class_name', 'accepted_count',
           'correct_count', 'pseudo_accuracy', 'scanned_count')


class PseudoLabelMetrics:
    """Count each target ID once per epoch; ground truth never leaves this object.

    Accuracy is a fraction in [0, 1], grouped by PREDICTED label. An empty
    selection has undefined accuracy (blank in CSV, N/A in console).
    """

    def __init__(self, true_labels):
        labels = np.asarray(true_labels)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('true_labels must be a one-dimensional integer array')
        if ((labels < 0) | (labels >= len(CLASS_NAMES))).any():
            raise ValueError('true_labels must use CAST class IDs 0..6')
        self._labels = labels.copy()
        self._seen = np.zeros(len(labels), dtype=bool)
        self.accepted = {s: np.zeros(7, dtype=np.int64) for s in STAGES}
        self.correct = {s: np.zeros(7, dtype=np.int64) for s in STAGES}

    def update(self, sample_ids, pseudo_labels, masks):
        ids, predictions = np.asarray(sample_ids), np.asarray(pseudo_labels)
        if (ids.ndim != 1 or predictions.shape != ids.shape
                or not np.issubdtype(ids.dtype, np.integer)
                or not np.issubdtype(predictions.dtype, np.integer)):
            raise ValueError('IDs and predictions must be matching integer vectors')
        if ((ids < 0) | (ids >= len(self._labels))).any():
            raise ValueError('target sample ID out of range')
        if len(np.unique(ids)) != len(ids) or self._seen[ids].any():
            raise ValueError('duplicate target IDs in epoch metrics')
        if ((predictions < 0) | (predictions >= 7)).any():
            raise ValueError('predictions must use CAST class IDs 0..6')
        if set(masks) != set(STAGES):
            raise ValueError('all diagnostic stage masks are required')
        checked = {}
        for stage, mask in masks.items():
            mask = np.asarray(mask)
            if mask.shape != ids.shape or mask.dtype != np.bool_:
                raise ValueError('stage masks must be matching boolean vectors')
            checked[stage] = mask
        # Validate everything before changing counters.
        self._seen[ids] = True
        is_correct = predictions == self._labels[ids]
        for stage, mask in checked.items():
            self.accepted[stage] += np.bincount(predictions[mask], minlength=7)
            self.correct[stage] += np.bincount(
                predictions[mask & is_correct], minlength=7)

    def write_epoch(self, path, epoch):
        if not self._seen.all():
            raise ValueError('epoch metrics require one visit to every target image')
        rows = []
        for stage in STAGES:
            counts, correct = self.accepted[stage], self.correct[stage]
            for c, name in enumerate(CLASS_NAMES):
                n, k = int(counts[c]), int(correct[c])
                accuracy = k / n if n else ''
                rows.append((epoch, stage, c, name, n, k, accuracy, len(self._labels)))
                text = '%.2f%%' % (100 * accuracy) if n else 'N/A'
                print('[Target Epoch %d] Pseudo_Quality stage=%s class=%s '
                      'accepted=%d correct=%d accuracy=%s' %
                      (epoch, stage, name, n, k, text))
            n, k = int(counts.sum()), int(correct.sum())
            accuracy = k / n if n else ''
            rows.append((epoch, stage, -1, 'ALL', n, k, accuracy, len(self._labels)))
            text = '%.2f%%' % (100 * accuracy) if n else 'N/A'
            print('[Target Epoch %d] Pseudo_Quality stage=%s class=ALL '
                  'accepted=%d correct=%d accuracy=%s' % (epoch, stage, n, k, text))
        with open(path, 'a', newline='') as handle:
            csv.writer(handle).writerows(rows)


def initialize_metrics_csv(path):
    with open(path, 'w', newline='') as handle:
        csv.writer(handle).writerow(COLUMNS)
