"""Offline RetinaFace detection + five-landmark similarity alignment to 256x256.

Keep raw data unchanged. No undetected image is silently used as an aligned face.
Run in an isolated preprocessing environment; training does not import TensorFlow.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from jaffe_manifest import scan_jaffe

# Standard five-point face template in 112 coordinates, scaled to output size.
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def similarity_transform(points, target):
    """Least-squares orientation-preserving similarity; no random estimator."""
    src, dst = np.asarray(points, dtype=float), np.asarray(target, dtype=float)
    if src.shape != (5, 2) or dst.shape != (5, 2) or not np.isfinite(src).all():
        raise ValueError('Invalid five-point landmarks')
    a = np.zeros((10, 4))
    a[0::2, 0], a[0::2, 1], a[0::2, 2] = src[:, 0], -src[:, 1], 1
    a[1::2, 0], a[1::2, 1], a[1::2, 3] = src[:, 1], src[:, 0], 1
    params, _, rank, _ = np.linalg.lstsq(a, dst.reshape(-1), rcond=None)
    if rank < 4 or np.hypot(*params[:2]) < 1e-8:
        raise ValueError('Degenerate facial landmarks')
    scale, rotation, tx, ty = params
    return np.array([[scale, -rotation, tx], [rotation, scale, ty]], dtype=np.float64)


def choose_face(detections):
    if not isinstance(detections, dict) or not detections:
        raise ValueError('No face detected')
    # JAFFE is a single-poser dataset; use largest face, then score, deterministically.
    def key(face):
        x1, y1, x2, y2 = face['facial_area']
        return (max(x2-x1, 0)*max(y2-y1, 0), float(face['score']))
    return max(detections.values(), key=key)


def align_face(image, face, size=256):
    import cv2
    lm = face['landmarks']
    eyes = sorted([lm['right_eye'], lm['left_eye']], key=lambda p: p[0])
    mouth = sorted([lm['mouth_right'], lm['mouth_left']], key=lambda p: p[0])
    points = np.asarray([*eyes, lm['nose'], *mouth], dtype=np.float64)
    matrix = similarity_transform(points, TEMPLATE * (size / 112.0))
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)), points


def run(args):
    import cv2
    from retinaface import RetinaFace
    root, out = Path(args.input).resolve(), Path(args.output).resolve()
    if out == root or root in out.parents or out in root.parents:
        raise ValueError('Input and output must be separate non-nested directories')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output must be empty; use a new directory')
    records = scan_jaffe(root)
    out.mkdir(parents=True, exist_ok=True)
    model = RetinaFace.build_model()
    rows = []
    for file, _ in records:
        row = {'input': file, 'output': '', 'status': 'failed', 'score': '',
               'faces': '', 'landmarks': '', 'error': ''}
        try:
            image = cv2.imread(file, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Image decoding failed')
            detections = RetinaFace.detect_faces(image, threshold=args.threshold, model=model)
            face = choose_face(detections)
            aligned, points = align_face(image, face)
            dest = out / Path(file).relative_to(root).with_suffix('.png')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dest), aligned):
                raise IOError('Image encoding failed')
            row.update(output=str(dest), status='ok', score=float(face['score']),
                       faces=len(detections), landmarks=json.dumps(points.tolist()))
        except Exception as error:
            row['error'] = str(error)
        rows.append(row)
        print('%s: %s %s' % (Path(file).name, row['status'], row['error']), flush=True)
    with (out / 'alignment_report.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    failed = sum(r['status'] != 'ok' for r in rows)
    (out / 'alignment_metadata.json').write_text(json.dumps(dict(
        method='RetinaFace five-point similarity', image_size=256,
        template_112=TEMPLATE.tolist(), threshold=args.threshold,
        input_count=len(rows), aligned_count=len(rows)-failed,
        failed_count=failed, input_dir=str(root)), indent=2), encoding='utf-8')
    if failed:
        raise RuntimeError('%d faces failed: inspect alignment_report.csv; do not train on a partial pool' % failed)
    print('Aligned %d images: %s' % (len(rows), out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float, default=0.9)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error('threshold must be in (0, 1]')
    run(args)
