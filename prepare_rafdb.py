"""Align RAF-DB train+test with exactly the same functions/template as JAFFE.

Create a separate dataset root, retain original label file and split membership.
No failed input is dropped or replaced with an unaligned fallback. --resume reuses
successful records of this same job to support long CPU preprocessing runs.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path
from prepare_jaffe import TEMPLATE, align_face, choose_face
from raf_alignment import read_raf_labels, choose_input_directory, find_input, aligned_raf_path


def run(args):
    import cv2
    from retinaface import RetinaFace
    root, out = Path(args.input).resolve(), Path(args.output).resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('Input and output must be separate non-nested directories')
    mode, folder = choose_input_directory(root, args.input_kind)
    records = read_raf_labels(root)
    # Fail before detection if there is any missing/ambiguous source image.
    inputs = [(name, label, find_input(folder, name, mode)) for name, label in records]
    job = dict(input_dir=str(root), input_kind=mode, threshold=args.threshold,
               method='RetinaFace five-point similarity', image_size=256,
               template_112=TEMPLATE.tolist(), label_records=records)
    # JSON roundtrip normalizes tuple/list types before resume comparison.
    job = json.loads(json.dumps(job))
    job_path = out / 'alignment_job.json'
    progress = out / 'alignment_progress.jsonl'
    if out.exists() and any(out.iterdir()):
        if not args.resume or not job_path.is_file():
            raise FileExistsError('Output nonempty: use --resume for the same job or a new directory')
        if json.loads(job_path.read_text()) != job:
            raise ValueError('Resume settings/label manifest differ from original job')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'Image/aligned').mkdir(parents=True, exist_ok=True)
    (out / 'EmoLabel').mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(job, indent=2), encoding='utf-8')
    shutil.copyfile(root / 'EmoLabel/list_patition_label.txt', out / 'EmoLabel/list_patition_label.txt')
    # Mark incomplete until the entire original manifest has succeeded.
    metadata_path = out / 'alignment_metadata.json'
    metadata = dict(method=job['method'], image_size=256,
                    template_112=TEMPLATE.tolist(), threshold=args.threshold,
                    input_count=len(records), aligned_count=0, failed_count=len(records),
                    input_dir=str(root), input_kind=mode, output_format='raf_aligned_png')
    metadata_path.write_text(json.dumps(metadata, indent=2))
    history = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            try:
                row = json.loads(line)
                history[row['name']] = row
            except (ValueError, KeyError):
                continue  # A killed process can leave a partial final record.
    model = None
    rows = []
    with progress.open('a', encoding='utf-8') as progress_file:
        # Separate a possibly truncated last JSON record from subsequent writes.
        if progress.stat().st_size:
            progress_file.write('\n')
        for index, (name, label, file) in enumerate(inputs, 1):
            dest = aligned_raf_path(out, name)
            previous = history.get(name)
            if previous and previous['status'] == 'ok' and dest.is_file():
                cached = cv2.imread(str(dest), cv2.IMREAD_COLOR)
                if cached is not None and cached.shape[:2] == (256, 256):
                    rows.append(previous)
                    continue
            row = dict(name=name, label=label, input=str(file), output=str(dest),
                       status='failed', score='', faces='', landmarks='', error='')
            try:
                if model is None:
                    model = RetinaFace.build_model()
                image = cv2.imread(str(file), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('Image decoding failed')
                detections = RetinaFace.detect_faces(image, threshold=args.threshold, model=model)
                face = choose_face(detections)
                aligned, points = align_face(image, face)
                if not cv2.imwrite(str(dest), aligned):
                    raise IOError('Image encoding failed')
                row.update(status='ok', score=float(face['score']), faces=len(detections),
                           landmarks=json.dumps(points.tolist()))
            except Exception as error:
                row['error'] = str(error)
            progress_file.write(json.dumps(row) + '\n')
            progress_file.flush()
            rows.append(row)
            print('[%d/%d] %s: %s %s' % (index, len(inputs), name, row['status'], row['error']), flush=True)
    with (out / 'alignment_report.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    failed = sum(r['status'] != 'ok' for r in rows)
    metadata.update(aligned_count=len(rows)-failed, failed_count=failed)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    if failed:
        raise RuntimeError('%d RAF images failed; inspect report and rerun --resume after addressing failures; no samples dropped' % failed)
    print('Aligned %d RAF images: %s; input_kind=%s' % (len(rows), out, mode))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--input_kind', choices=['auto', 'original', 'aligned'], default='auto')
    p.add_argument('--threshold', type=float, default=.9)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not 0 < args.threshold <= 1:
        p.error('threshold must be in (0,1]')
    run(args)
