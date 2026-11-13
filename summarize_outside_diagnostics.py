"""Offline only: python summarize_outside_diagnostics.py RUN/epoch_030.csv"""
import csv
import sys
from collections import defaultdict


def summarize(path):
    counts = defaultdict(lambda: [0, 0])
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        has_truth = 'correct' in reader.fieldnames
        for row in reader:
            region = row['region']
            groups = [region]
            if region == 'outside' and row['knn_checked'] == 'True' and row['knn_pass'] == 'True':
                groups.append('outside_knn_pass')
            for group in groups:
                keys = [('all', 'all', group), ('pred', row['pseudo_label'], group)]
                if has_truth:
                    keys.append(('true', row['true_label'], group))
                for key in keys:
                    counts[key][0] += 1
                    counts[key][1] += int(row['correct']) if has_truth else 0
    print('grouping,class,region,count,correct,accuracy')
    for key, (n, correct) in sorted(counts.items()):
        print(','.join(key) + ',%d,%s,%s' % (
            n, correct if has_truth else 'NA', '%.4f' % (correct/n) if has_truth else 'NA'))


if __name__ == '__main__':
    summarize(sys.argv[1])
