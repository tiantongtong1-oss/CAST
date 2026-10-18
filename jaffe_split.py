"""Fixed class-stratified JAFFE manifests; labels used only during split creation."""
import argparse
import json
import random
from pathlib import Path
from jaffe_manifest import scan_jaffe


def create_split(root, seed=1314):
    root = Path(root)
    records = scan_jaffe(root)
    counts = [sum(y == c for _, y in records) for c in range(7)]
    if counts != [30, 32, 29, 31, 31, 30, 30]:
        raise ValueError('Expected complete canonical 213-image JAFFE pool; counts=%s' % counts)
    rng = random.Random(seed)
    train, test = [], []
    for c, count in enumerate([6, 7, 6, 6, 6, 6, 6]):
        paths = [Path(p).relative_to(root).as_posix() for p, y in records if y == c]
        rng.shuffle(paths)
        test.extend(paths[:count])
        train.extend(paths[count:])
    return dict(version=1, protocol='jaffe_stratified_170_43', seed=seed,
                train=sorted(train), test=sorted(test))


def select_split(root, records, manifest, phase):
    root = Path(root)
    obj = json.loads(Path(manifest).read_text())
    train, test = obj['train'], obj['test']
    if (obj.get('protocol') != 'jaffe_stratified_170_43' or
            len(train) != 170 or len(test) != 43 or
            len(set(train + test)) != 213):
        raise ValueError('Invalid JAFFE 170/43 manifest or overlapping/duplicate files')
    by_path = {Path(p).relative_to(root).as_posix(): (p, y) for p, y in records}
    if set(train + test) != set(by_path):
        raise ValueError('JAFFE manifest does not match dataset files')
    selected = train if phase == 'adapt' else test
    return [by_path[p] for p in sorted(selected)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=1314)
    args = parser.parse_args()
    obj = create_split(args.root, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')
    print('Saved %s: train=170, test=43, seed=%d' % (out, args.seed))


if __name__ == '__main__':
    main()
