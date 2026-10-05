"""Read-only prediction diagnostics; no target labels enter training decisions."""
import numpy as np

CLASS_NAMES = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']


def target_training_mask(final_mask, pseudo_targets, exclude_disgust=False):
    mask = final_mask.clone()
    if exclude_disgust:
        mask = mask & pseudo_targets.ne(2)
    return mask


class PredictionAudit:
    """Accumulate small CPU confusion matrices instead of retaining batches."""
    def __init__(self):
        self.matrices = {}
        self.total = 0
        self.true_counts = np.zeros(7, dtype=np.int64)

    def add(self, targets, predictions, masks=None):
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
        if targets.shape != predictions.shape:
            raise ValueError('targets and predictions must have the same shape')
        if np.any((targets < 0) | (targets > 6) | (predictions < 0) | (predictions > 6)):
            raise ValueError('class IDs must be in [0, 6]')
        self.total += targets.size
        self.true_counts += np.bincount(targets, minlength=7)
        groups = {'all': np.ones(targets.size, dtype=bool)}
        groups.update(masks or {})
        for name, mask in groups.items():
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if mask.shape != targets.shape:
                raise ValueError('mask must match targets')
            cm = np.bincount(targets[mask] * 7 + predictions[mask], minlength=49).reshape(7, 7)
            if name not in self.matrices:
                self.matrices[name] = np.zeros((7, 7), dtype=np.int64)
            self.matrices[name] += cm

    def report(self, name):
        print('\n[PREDICTION AUDIT] %s total=%d' % (name, self.total))
        print('class order:', CLASS_NAMES)
        for group, cm in self.matrices.items():
            count = int(cm.sum())
            tp = cm.diagonal().astype(float)
            predicted = cm.sum(axis=0)
            selected_true = cm.sum(axis=1)
            precision = np.divide(tp, predicted, out=np.zeros(7), where=predicted > 0)
            recall = np.divide(tp, selected_true, out=np.zeros(7), where=selected_true > 0)
            # For selected groups: correct accepted samples / ALL true samples.
            accepted_recall = np.divide(tp, self.true_counts, out=np.zeros(7), where=self.true_counts > 0)
            f1 = np.divide(2 * precision * recall, precision + recall,
                           out=np.zeros(7), where=precision + recall > 0)
            overall = '%.4f' % (tp.sum() / count) if count else 'NA'
            print('[%s] count=%d coverage=%.4f precision=%s wrong=%d' %
                  (group, count, count / max(self.total, 1), overall, count - tp.sum()))
            print('rows=true, cols=pred; raw counts:\n', cm)
            for c, label in enumerate(CLASS_NAMES):
                p = '%.4f' % precision[c] if predicted[c] else 'NA'
                print('  %d %s: accepted=%d correct=%d precision=%s '
                      'selected_recall=%.4f accepted_recall=%.4f' %
                      (c, label, predicted[c], tp[c], p, recall[c], accepted_recall[c]))
            print('macro-F1%s=%.4f' % (' (selected subset)' if group != 'all' else '', f1.mean()))
        print('[END PREDICTION AUDIT]\n')
