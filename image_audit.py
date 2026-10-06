"""Export original FER images for read-only checkpoint diagnosis."""
import csv
import json
import random
import shutil
from pathlib import Path
import numpy as np
from source_knn_audit import NAMES


class ImageAudit:
    GROUPS = {(5, 2): 'angry_to_disgust', (2, 2): 'disgust_correct',
              (2, 4): 'disgust_to_sad', (2, 5): 'disgust_to_angry'}

    def __init__(self, paths, per_subset=30):
        self.paths = paths
        self.offset = 0
        self.per_subset = per_subset
        self.neighbors = {name: [] for name in self.GROUPS.values()}
        self.rows = {name: [] for name in self.GROUPS.values()}

    def add(self, targets, predictions, confidences, accepted,
            probabilities=None, diagnostics=None, neighbors=None):
        n = len(targets)
        if not all(len(values) == n for values in (predictions, confidences, accepted)):
            raise ValueError('Image audit batch lengths differ')
        for extra in (probabilities, diagnostics, neighbors):
            if extra is not None and len(extra) != n:
                raise ValueError('diagnostics must align with batch')
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
                row = self.rows[group][-1]
                if probabilities is not None:
                    row.update({'p_' + name: float(probabilities[j][c])
                                for c, name in enumerate(NAMES)})
                if diagnostics is not None:
                    row.update(diagnostics[j])
                if neighbors is not None:
                    self.neighbors[group].extend(
                        dict(target_sample_id=self.offset + j, **neighbor)
                        for neighbor in neighbors[j])
        self.offset += n

    def save(self, directory, checkpoint, source_metadata=None):
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
            group_fields = fields + sorted({key for row in rows for key in row} - set(fields))
            rows = sorted(rows, key=lambda r: (-r['confidence'], r['sample_id']))
            with (folder / 'samples.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=group_fields)
                writer.writeheader()
                writer.writerows(rows)
            if source_metadata is not None:
                with (folder / 'neighbors.csv').open('w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'target_sample_id', 'rank', 'source_path', 'source_true', 'cosine'])
                    writer.writeheader()
                    writer.writerows(self.neighbors[name])
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
        metadata['source_knn'] = source_metadata
        summary = {}
        for name, rows in self.rows.items():
            stats = dict(count=len(rows))
            if rows and source_metadata is not None:
                for prefix in ('source', 'balanced_source'):
                    for c in (2, 5):
                        stats[prefix + '_predict_' + NAMES[c] + '_fraction'] = float(
                            np.mean([r[prefix + '_knn_pred'] == c for r in rows]))
                    stats[prefix + '_agreement_fraction'] = float(np.mean([
                        r[prefix + '_knn_agrees'] for r in rows]))
                    stats[prefix + '_median_disgust_support'] = float(np.median([
                        r[prefix + '_support_disgust'] for r in rows]))
            summary[name] = stats
        (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        (root / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('[IMAGE AUDIT] saved to %s' % root.resolve(), flush=True)
