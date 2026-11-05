"""Binary classification metrics with explicit sample-weighted loss."""
import math


def from_confusion(cm):
    if len(cm) != 2 or any(len(row) != 2 for row in cm):
        raise ValueError('Expected a 2x2 confusion matrix')
    if any(not isinstance(n, int) or n < 0 for row in cm for n in row):
        raise ValueError('Counts must be nonnegative integers')
    total = sum(map(sum, cm))
    if total == 0:
        raise ValueError('Cannot evaluate an empty dataset')
    classes = {}
    for i, name in enumerate(('negative', 'positive')):
        tp, fp, fn = cm[i][i], cm[1-i][i], cm[i][1-i]
        classes[name] = {
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'support': sum(cm[i]),
        }
    return {'samples': total, 'accuracy': (cm[0][0] + cm[1][1]) / total,
            'macro_f1': sum(row['f1'] for row in classes.values()) / 2,
            'confusion_matrix': [list(row) for row in cm], 'classes': classes}


class BinaryMetrics:
    def __init__(self):
        self.cm = [[0, 0], [0, 0]]
        self.loss_sum = 0.0

    def update(self, labels, predictions, loss_sum):
        if len(labels) != len(predictions) or not math.isfinite(loss_sum):
            raise ValueError('Invalid batch length or nonfinite loss')
        if any(y not in (0, 1) for y in list(labels) + list(predictions)):
            raise ValueError('Labels and predictions must be binary')
        for label, pred in zip(labels, predictions):
            self.cm[int(label)][int(pred)] += 1
        self.loss_sum += loss_sum

    def result(self):
        result = from_confusion(self.cm)
        result['loss'] = self.loss_sum / result['samples']
        return result
