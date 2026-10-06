"""Export original FER images for read-only checkpoint diagnosis."""
import csv
import json
import random
import shutil
from pathlib import Path


class ImageAudit:
    GROUPS = {(5, 2): 'angry_to_disgust', (2, 2): 'disgust_correct',
              (2, 4): 'disgust_to_sad', (2, 5): 'disgust_to_angry'}

    def __init__(self, paths, per_subset=30):
        self.paths = paths
        self.offset = 0
        self.per_subset = per_subset
        self.rows = {name: [] for name in self.GROUPS.values()}

    def add(self, targets, predictions, confidences, accepted):
        n = len(targets)
        if not all(len(values) == n for values in (predictions, confidences, accepted)):
            raise ValueError('Image audit batch lengths differ')
        if self.offset + n > len(self.paths):
            raise ValueError('Image audit exceeds dataset paths')
        for j, (true, pred, conf, passed) in enumerate(zip(
                targets, predictions, confidences, accepted)):
            group = self.GROUPS.get((int(true), int(pred)))
            if group:
                self.rows[group].append(dict(
                    sample_id=self.offset + j,
                    path=str(Path(self.paths[self.offset + j]).resolve()),
                    true=int(true), pred=int(pred), confidence=float(conf),
                    confidence_accepted=bool(passed)))
        self.offset += n

    def save(self, directory, checkpoint):
        if self.offset != len(self.paths):
            raise ValueError('Image audit did not cover the complete dataset')
        root = Path(directory)
        # Refuse reuse so old images cannot contaminate a new comparison.
        root.mkdir(parents=True, exist_ok=False)
        fields = ['sample_id', 'path', 'true', 'pred', 'confidence',
                  'confidence_accepted']
        for name, rows in self.rows.items():
            folder = root / name
            folder.mkdir()
            rows = sorted(rows, key=lambda r: (-r['confidence'], r['sample_id']))
            with (folder / 'samples.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            high = rows[:self.per_subset]
            rest = rows[self.per_subset:]
            selected = random.Random(2000).sample(rest, min(self.per_subset, len(rest)))
            for subset, samples in [('high_conf', high), ('random', selected)]:
                dest = folder / subset
                dest.mkdir()
                for row in samples:
                    source = Path(row['path'])
                    filename = '%06d_p%.4f_%s' % (
                        row['sample_id'], row['confidence'], source.name)
                    shutil.copy2(source, dest / filename)
            print('[IMAGE AUDIT] %s: total=%d exported=%d' %
                  (name, len(rows), len(high) + len(selected)), flush=True)
        metadata = dict(checkpoint=str(Path(checkpoint).resolve()), split='train',
                        view='resize 256 / center crop 224', model='student',
                        sample_count=self.offset, per_subset=self.per_subset,
                        class_order=['surprise', 'fear', 'disgust', 'happy',
                                     'sad', 'angry', 'neutral'])
        (root / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('[IMAGE AUDIT] saved to %s' % root.resolve(), flush=True)
