"""JAFFE filename labels (requested poses), not semantic rating argmax."""
from pathlib import Path
import re

POSE_TO_CAST = {'SU': 0, 'FE': 1, 'DI': 2, 'HA': 3, 'SA': 4, 'AN': 5, 'NE': 6}
IMAGE_SUFFIXES = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def scan_jaffe(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError('JAFFE directory does not exist: %s' % root)
    records = []
    seen = set()
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        # Original examples: KA.AN1.39.tiff; README pose IDs: KA-AN1.
        match = re.fullmatch(r'([A-Z]{2})[.\-_]([A-Z]{2})([1-5])(?:[.\-_]\d+)?', path.stem.upper())
        if not match or match[2] not in POSE_TO_CAST:
            raise ValueError('Unrecognized JAFFE filename: %s' % path)
        key = (match[1], match[2], match[3])
        if key in seen:
            raise ValueError('Duplicate JAFFE pose (possibly copied splits): %s' % path)
        seen.add(key)
        records.append((str(path), POSE_TO_CAST[match[2]]))
    if not records:
        raise FileNotFoundError('No JAFFE images found under %s' % root)
    return records
