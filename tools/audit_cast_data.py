#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from cast_v6.data import build_fer2013, build_rafdb


RAF_TRAIN_EXPECTED = [1290, 281, 717, 4772, 1982, 705, 2524]
FER_TRAIN_CAST_EXPECTED = [3171, 4097, 436, 7215, 4830, 3995, 4965]
FER_PUBLIC_CAST_EXPECTED = [415, 496, 56, 895, 653, 467, 607]
FER_PRIVATE_CAST_EXPECTED = [416, 528, 55, 879, 594, 491, 626]
CLASS_NAMES = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]


def md5_file(path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def inspect_dataset(name, ds, expected=None, decode_per_class=8):
    counts = ds.class_counts
    print("[%s] n=%d counts=%s" % (name, len(ds), counts))
    if expected is not None:
        print("[%s] expected=%s match=%s" % (name, expected, counts == expected))
    sizes = Counter()
    bad = []
    examples = {c: [] for c in range(7)}
    seen = [0] * 7
    for path, label in zip(ds.paths, ds.labels):
        if seen[label] >= decode_per_class:
            continue
        try:
            with Image.open(path) as im:
                sizes[im.size] += 1
            examples[label].append(path)
        except Exception as exc:
            bad.append((path, repr(exc)))
        seen[label] += 1
        if all(v >= decode_per_class for v in seen):
            break
    print("[%s] sampled_sizes=%s decode_errors=%d" % (name, dict(sizes), len(bad)))
    for c in range(7):
        print("[%s] sample_paths[%d:%s]=%s" % (name, c, CLASS_NAMES[c], examples[c][:3]))
    return {"counts": counts, "sizes": {str(k): v for k, v in sizes.items()}, "bad": bad}


def exact_duplicate_count(ds_a, ds_b):
    hashes = {}
    for p in ds_a.paths:
        hashes.setdefault(md5_file(p), p)
    dup = []
    for p in ds_b.paths:
        h = md5_file(p)
        if h in hashes:
            dup.append((hashes[h], p))
    return dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--target-root", required=True)
    ap.add_argument("--fer-folder-order", default="cast", choices=["cast", "kaggle"])
    ap.add_argument("--hash-duplicates", action="store_true")
    args = ap.parse_args()

    raf_train = build_rafdb(args.source_root, "train", mode="eval")
    raf_test = build_rafdb(args.source_root, "test", mode="eval")
    fer_train = build_fer2013(args.target_root, "train", mode="eval", folder_order=args.fer_folder_order)
    fer_test = build_fer2013(args.target_root, "test", mode="eval", folder_order=args.fer_folder_order)
    try:
        fer_val = build_fer2013(args.target_root, "val", mode="eval", folder_order=args.fer_folder_order)
    except FileNotFoundError:
        fer_val = None

    report = {}
    report["raf_train"] = inspect_dataset("RAF train", raf_train, RAF_TRAIN_EXPECTED)
    report["raf_test"] = inspect_dataset("RAF test", raf_test)
    report["fer_train"] = inspect_dataset("FER train", fer_train, FER_TRAIN_CAST_EXPECTED)
    if fer_val is not None:
        report["fer_val"] = inspect_dataset("FER val", fer_val, FER_PUBLIC_CAST_EXPECTED)
    else:
        print("[FER val][WARN] missing. Paper protocol uses 28,709 train + 3,589 validation + 3,589 test.")
    report["fer_test"] = inspect_dataset("FER test", fer_test, FER_PRIVATE_CAST_EXPECTED)

    if args.hash_duplicates:
        dup = exact_duplicate_count(fer_train, fer_test)
        print("[Leakage] exact FER train/test duplicates=%d" % len(dup))
        report["fer_train_test_duplicates"] = dup[:50]

    if fer_train.class_counts == FER_TRAIN_CAST_EXPECTED:
        print("[Label-map] FER train count signature matches CAST order: surprise,fear,disgust,happy,sad,angry,neutral.")
    else:
        print("[Label-map][WARN] FER class count signature differs from the expected CAST-order FER2013 train split.")

    print("\nManual semantic check is still required: open several sample_paths from every class; counts cannot prove folder semantics.")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
