"""Offline summary of independent candidates; labels used for evaluation only."""
import csv
import json
import sys


def summarize(path):
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    lines = ['Scope: all target TRAIN samples, deterministic single view.',
             'Class: 0 surprise, 1 fear, 2 disgust, 3 happy, 4 sad, 5 angry, 6 neutral.',
             'class,n,teacher_acc,teacher_top2,teacher_top3,medoid_acc,source_knn_acc,balanced_knn_acc,union_teacher_source_top2']
    def mean(group, test):
        return '%.4f' % (sum(test(r) for r in group) / len(group)) if group else 'NA'
    for c in ['all'] + list(range(7)):
        group = rows if c == 'all' else [r for r in rows if int(r['true_label']) == c]
        y = lambda r: int(r['true_label'])
        metrics = [lambda r: int(r['teacher_pred']) == y(r),
                   lambda r: y(r) in json.loads(r['teacher_top2']),
                   lambda r: y(r) in json.loads(r['teacher_top3']),
                   lambda r: int(r['medoid_pred']) == y(r),
                   lambda r: int(r['source_knn_pred']) == y(r),
                   lambda r: int(r['balanced_knn_pred']) == y(r),
                   lambda r: y(r) in set(json.loads(r['teacher_top2']) + json.loads(r['source_knn_top2']))]
        lines.append(','.join([str(c), str(len(group))] + [mean(group, m) for m in metrics]))
    lines += ['', 'Potential replacement outcomes; fixed methods, not a deployable correction rule.',
              'class,method,disagree,teacher_wrong_alternative_right,teacher_right_alternative_wrong,both_wrong']
    for c in ['all'] + list(range(7)):
        group = rows if c == 'all' else [r for r in rows if int(r['true_label']) == c]
        for method in ['medoid_pred', 'source_knn_pred', 'balanced_knn_pred']:
            disagree = [r for r in group if r[method] != r['teacher_pred']]
            gain = sum(r['teacher_pred'] != r['true_label'] and r[method] == r['true_label'] for r in disagree)
            harm = sum(r['teacher_pred'] == r['true_label'] and r[method] != r['true_label'] for r in disagree)
            lines.append('%s,%s,%d,%d,%d,%d' % (c, method, len(disagree), gain, harm, len(disagree)-gain-harm))
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    print(summarize(sys.argv[1]), end='')
