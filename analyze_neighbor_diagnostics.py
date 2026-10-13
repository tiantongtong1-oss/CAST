"""Read-only CSV analysis; never feeds decisions back into training."""
import argparse
import csv
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    args = parser.parse_args()
    with open(args.csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print('No radius-rescue candidates recorded.'); return
    print('Rows:', len(rows), '(sample-epoch observations, not unique samples)')
    for key in ('support_before', 'support_after', 'local_margin', 'largest_group_fraction'):
        values = np.array([float(r[key]) for r in rows])
        values = values[np.isfinite(values)]
        print(key, 'valid:', len(values), 'q0/q25/q50/q75/q100:',
              np.quantile(values, [0, .25, .5, .75, 1]).round(4) if len(values) else [])
    paired = [(float(r['support_before']), float(r['support_after'])) for r in rows
              if np.isfinite(float(r['support_before'])) and np.isfinite(float(r['support_after']))]
    if paired:
        delta = np.array([b-a for a,b in paired])
        print('Exclusion changed support fraction:', np.mean(np.abs(delta) > 1e-6))
    if 'pseudo_correct' not in rows[0]:
        print('No true-label audit: cannot infer pseudo-label correctness from support alone.'); return
    print('Privileged diagnostic labels: not an unsupervised metric or threshold guarantee.')
    for label in sorted(set(int(r['pseudo_label']) for r in rows)):
        subset = [r for r in rows if int(r['pseudo_label']) == label]
        passed = [r for r in subset if int(r['would_pass'])]
        acc = lambda rs: np.mean([int(r['pseudo_correct']) for r in rs]) if rs else float('nan')
        print('class', label, 'candidate_count', len(subset), 'accuracy', acc(subset),
              'would_pass_count', len(passed), 'accuracy', acc(passed))
    for correct in (0, 1):
        subset = [r for r in rows if int(r['pseudo_correct']) == correct]
        print('correct=', correct, 'count=', len(subset))
        for key in ('confidence', 'source_distance', 'support_before', 'support_after', 'local_margin'):
            vals = np.array([float(r[key]) for r in subset]); vals = vals[np.isfinite(vals)]
            print(' ', key, 'median=', float(np.median(vals)) if len(vals) else float('nan'))


if __name__ == '__main__':
    main()
