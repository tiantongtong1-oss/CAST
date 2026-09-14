from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from randaugment import RandAugmentMC  # original CAST repo helper
except Exception:  # pragma: no cover
    RandAugmentMC = None


CAST_CLASS_NAMES = ("surprise", "fear", "disgust", "happy", "sad", "angry", "neutral")
KAGGLE_TO_CAST = {0: 5, 1: 2, 2: 1, 3: 3, 4: 4, 5: 0, 6: 6}


def build_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    weak = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    strong_ops = [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
    ]
    if RandAugmentMC is not None:
        strong_ops.append(RandAugmentMC(n=2, m=10))
    elif hasattr(transforms, "RandAugment"):
        strong_ops.append(transforms.RandAugment(num_ops=2, magnitude=10))
    strong_ops.extend([
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
    strong = transforms.Compose(strong_ops)

    test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return weak, strong, test


class MultiViewDataset(Dataset):
    """Dataset that returns two weak views and one strong view from the same raw image."""

    def __init__(self, paths: Sequence[str], labels: Sequence[int], mode: str = "train",
                 weak_transform: Callable | None = None,
                 strong_transform: Callable | None = None,
                 test_transform: Callable | None = None):
        self.paths = list(paths)
        self.labels = [int(x) for x in labels]
        self.mode = mode
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.test_transform = test_transform

    def __len__(self):
        return len(self.paths)

    def _load(self, index: int) -> Image.Image:
        return Image.open(self.paths[index]).convert("RGB")

    def __getitem__(self, index: int):
        image = self._load(index)
        label = self.labels[index]
        if self.mode == "train":
            w1 = self.weak_transform(image)
            w2 = self.weak_transform(image)
            strong = self.strong_transform(image)
            return w1, w2, strong, label, index
        if self.mode == "pseudo":
            w1 = self.weak_transform(image)
            w2 = self.weak_transform(image)
            return w1, w2, label, index
        x = self.test_transform(image)
        return x, label, index

    @property
    def class_counts(self) -> List[int]:
        arr = np.asarray(self.labels)
        return [int(np.sum(arr == c)) for c in range(7)]


def _raf_paths_labels(root: str, split: str) -> Tuple[List[str], List[int]]:
    label_file = Path(root) / "EmoLabel" / "list_patition_label.txt"
    if not label_file.exists():
        raise FileNotFoundError("RAF-DB label file not found: %s" % label_file)
    paths, labels = [], []
    prefix = "train" if split == "train" else "test"
    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            name, label = line.strip().split()
            if not name.startswith(prefix):
                continue
            aligned = name.rsplit(".", 1)[0] + "_aligned.jpg"
            paths.append(str(Path(root) / "Image" / "aligned" / aligned))
            labels.append(int(label) - 1)
    return paths, labels


def build_rafdb(root: str, split: str, mode: str | None = None) -> MultiViewDataset:
    weak, strong, test = build_transforms()
    paths, labels = _raf_paths_labels(root, split)
    if mode is None:
        mode = "train" if split == "train" else "eval"
    return MultiViewDataset(paths, labels, mode=mode,
                            weak_transform=weak, strong_transform=strong,
                            test_transform=test)


def resolve_fer_split(root: str, split: str) -> str:
    candidates = {
        "train": ["train", "Train", "training"],
        "val": ["val", "valid", "validation", "publictest", "PublicTest"],
        "test": ["test", "Test", "private_test", "privatetest", "PrivateTest"],
    }[split]
    for name in candidates:
        p = Path(root) / name
        if p.is_dir():
            return str(p)
    raise FileNotFoundError(
        "FER2013 split '%s' not found under %s. Tried: %s"
        % (split, root, ", ".join(candidates))
    )


def _fer_paths_labels(root: str, split: str, folder_order: str) -> Tuple[List[str], List[int]]:
    split_dir = resolve_fer_split(root, split)
    paths, labels = [], []
    for class_dir in sorted(Path(split_dir).iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            raw_label = int(class_dir.name)
        except ValueError:
            lower = class_dir.name.lower()
            aliases = {
                "surprise": 0, "fear": 1, "disgust": 2, "happy": 3,
                "happiness": 3, "sad": 4, "sadness": 4, "angry": 5,
                "anger": 5, "neutral": 6,
            }
            if lower not in aliases:
                continue
            raw_label = aliases[lower]
            folder_order = "cast"
        label = KAGGLE_TO_CAST[raw_label] if folder_order == "kaggle" else raw_label
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in class_dir.glob(ext):
                paths.append(str(p))
                labels.append(label)
    if not paths:
        raise RuntimeError("No FER2013 images found in %s" % split_dir)
    order = np.argsort(np.asarray(paths))
    paths = [paths[i] for i in order]
    labels = [labels[i] for i in order]
    return paths, labels


def build_fer2013(root: str, split: str, mode: str | None = None,
                   folder_order: str = "cast") -> MultiViewDataset:
    if folder_order not in ("cast", "kaggle"):
        raise ValueError("folder_order must be 'cast' or 'kaggle'")
    weak, strong, test = build_transforms()
    paths, labels = _fer_paths_labels(root, split, folder_order)
    if mode is None:
        mode = "train" if split == "train" else "eval"
    return MultiViewDataset(paths, labels, mode=mode,
                            weak_transform=weak, strong_transform=strong,
                            test_transform=test)


def print_dataset_summary(name: str, ds: MultiViewDataset) -> None:
    print("[%s] images=%d class_counts=%s" % (name, len(ds), ds.class_counts))
