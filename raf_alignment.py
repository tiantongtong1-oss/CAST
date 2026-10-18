"""RAF image naming and shared preprocessing provenance checks (no image dependencies)."""
import json
from pathlib import Path


def read_raf_labels(root):
    path = Path(root) / 'EmoLabel/list_patition_label.txt'
    records, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name, label = line.split()
        if Path(name).name != name or not name.startswith(('train_', 'test_')):
            raise ValueError('Invalid RAF filename: %s' % name)
        if name in seen or not 1 <= int(label) <= 7:
            raise ValueError('Duplicate RAF filename or invalid label: %s' % name)
        seen.add(name)
        records.append((name, int(label)))
    if not records:
        raise ValueError('Empty RAF label file')
    return records


def choose_input_directory(root, mode):
    root = Path(root)
    original, aligned = root / 'Image/original', root / 'Image/aligned'
    if mode == 'auto':
        mode = 'original' if original.is_dir() else 'aligned'
    folder = original if mode == 'original' else aligned
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return mode, folder


def find_input(folder, name, mode):
    stem = Path(name).stem + ('_aligned' if mode == 'aligned' else '')
    candidates = [Path(folder) / (stem + ext) for ext in ('.jpg', '.png', '.jpeg', '.bmp')]
    existing = [p for p in candidates if p.is_file()]
    if len(existing) != 1:
        raise ValueError('Expected exactly one RAF input for %s, found %d' % (name, len(existing)))
    return existing[0]


def aligned_raf_path(root, name):
    return Path(root) / 'Image/aligned' / (Path(name).stem + '_aligned.png')


def validate_shared_alignment(source_root, target_root):
    metas = []
    for root in (source_root, target_root):
        path = Path(root) / 'alignment_metadata.json'
        if not path.is_file():
            raise ValueError('Missing alignment metadata: %s' % path)
        meta = json.loads(path.read_text())
        if meta['failed_count'] or meta['input_count'] != meta['aligned_count']:
            raise ValueError('Incomplete alignment pool: %s' % root)
        metas.append(meta)
    for key in ('method', 'image_size', 'template_112', 'threshold'):
        if metas[0][key] != metas[1][key]:
            raise ValueError('Source/target alignment mismatch: %s' % key)
    return metas
