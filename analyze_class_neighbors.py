"""Epoch/class diagnostics with paired, equal-budget comparisons."""
import argparse
import csv
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv_path')
    p.add_argument('--fractions', type=float, nargs='+', default=[.25, .5, .75])
    args = p.parse_args()
    if any(not 0 < f <= 1 for f in args.fractions):
        p.error('fractions must be in (0,1]')
    with open(args.csv_path, newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        print('No rescue candidate observations'); return
    if 'source_margin' not in rows[0]:
        p.error('CSV requires --class_neighbor_diagnostics')
    metrics = ('confidence', 'source_distance', 'source_margin', 'target_margin')
    labeled = 'pseudo_correct' in rows[0]
    print('Only radius-rescue candidates; per epoch/class, repeated epochs are not independent.')
    print('Privileged labels used for diagnosis only.' if labeled else 'No true labels: correctness cannot be evaluated.')
    for epoch in sorted(set(int(r['epoch']) for r in rows)):
        for c in sorted(set(int(r['pseudo_label']) for r in rows)):
            group = [r for r in rows if int(r['epoch']) == epoch and int(r['pseudo_label']) == c]
            if not group:
                continue
            print('\nepoch', epoch, 'predicted_class', c, 'candidates', len(group))
            for domain in ('source', 'target'):
                vals = np.array([float(r[domain+'_margin']) for r in group])
                finite = np.isfinite(vals)
                print(domain, 'ready', int(finite.sum()), 'margin q10/q50/q90',
                      np.quantile(vals[finite], [.1,.5,.9]).round(4).tolist() if finite.any() else [])
                if labeled:
                    for correct in (0,1):
                        v = [float(r[domain+'_margin']) for r in group
                             if int(r['pseudo_correct']) == correct and np.isfinite(float(r[domain+'_margin']))]
                        print(' correct', correct, 'n', len(v), 'median', np.median(v) if v else 'missing')
            if not labeled:
                continue
            # Identical candidate pool for all four metrics. Report lost coverage.
            common = [r for r in group if all(np.isfinite(float(r[m])) for m in metrics)]
            print('common comparison pool', len(common), '/', len(group))
            if not common:
                continue
            common.sort(key=lambda r: int(r['sample_id']))  # deterministic tie-breaking
            for fraction in args.fractions:
                budget = max(1, int(len(common)*fraction))
                for metric in metrics:
                    selected = sorted(common, key=lambda r: float(r[metric]),
                                      reverse=metric != 'source_distance')[:budget]
                    precision = np.mean([int(r['pseudo_correct']) for r in selected])
                    print(' budget', budget, 'metric', metric, 'precision', round(float(precision),4))


if __name__ == '__main__':
    main()
