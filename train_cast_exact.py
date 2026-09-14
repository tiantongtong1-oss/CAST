"""Exact public-repository reproduction path for CAST (RAF-DB -> FER2013).

This file intentionally follows smwanghhh/CAST's public train.py + Networks.py
training behavior instead of the stabilized v6.x implementation.

Intentional reproduction choices:
- source pre-training: 30 epochs
- target adaptation: 30 epochs
- Adam lr=1e-3, weight_decay=1e-4, ExponentialLR gamma=0.95
- alpha/beta/gamma = 4 / 0.3 / 0.1, phi=1.4
- public-repo per-mini-batch CATM threshold rule
- public-repo MK-MMD implementation (5 kernels summed, 6 random matches)
- public-repo KDE/volume weighting implementation
- source + strong-target concatenated second forward
- one backward pass per target batch
- no EMA, no temperature calibration, no pseudo-bank, no keep-ratio,
  no source correction, no DDRL warmup, no BN freezing/recalibration

Compatibility-only changes relative to the public repository:
1) data roots are CLI arguments;
2) canonical FER2013 folder labels are remapped to CAST label semantics;
3) for a canonical FER2013 split, train + val are merged into the public
   repository's single target "train" set. This yields the paper Table-I
   FER2013 distribution [3586,4593,492,8110,5483,4462,5572] in CAST order;
4) old torchvision pretrained=True semantics are pinned to IMAGENET1K_V1
   rather than modern DEFAULT weights;
5) modern torch.load compatibility;
6) diagnostics are clearer, but do not alter optimization.

IMPORTANT PROTOCOL NOTE:
The public repository evaluates the target TEST set after every source and
adaptation epoch and uses it to select the best checkpoint. That is target-test
model-selection leakage. It is preserved by default here only to reproduce the
public repository. Use --selection-split val (and --no-combine-target-val) for
a cleaner protocol, but that is not repo-exact.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from randaugment import RandAugmentMC


warnings.filterwarnings("ignore")

CAST_CLASS_NAMES = ("surprise", "fear", "disgust", "happy", "sad", "angry", "neutral")
KAGGLE_TO_CAST = {0: 5, 1: 2, 2: 1, 3: 3, 4: 4, 5: 0, 6: 6}
PAPER_FER_TRAIN_COUNTS = [3586, 4593, 492, 8110, 5483, 4462, 5572]


def parse_args():
    p = argparse.ArgumentParser("CAST exact public-repository reproduction")
    p.add_argument("--source-root", required=True)
    p.add_argument("--target-root", required=True)
    p.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50", "mobilenet_v2"])
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-epochs", type=int, default=30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr-gamma", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--w1", type=float, default=4.0)
    p.add_argument("--w2", type=float, default=0.3)
    p.add_argument("--w3", type=float, default=0.1)
    p.add_argument("--phi", type=float, default=1.4)
    p.add_argument("--fer-folder-order", default="kaggle", choices=["kaggle", "cast"])
    p.add_argument("--selection-split", default="test", choices=["test", "val"],
                   help="repo-exact default is test; val is a cleaner non-exact option")
    p.add_argument("--no-combine-target-val", action="store_true",
                   help="do not merge canonical FER train+val for target training")
    p.add_argument("--no-imagenet-pretrained", action="store_true")
    p.add_argument("--checkpoint", default="",
                   help="optional original-format checkpoint; exact default is ImageNet init + 30 source epochs")
    p.add_argument("--model-dir", default="")
    return p.parse_args()


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _build_backbone(name: str, pretrained: bool):
    """Match the legacy torchvision pretrained=True initialization.

    The public CAST environment is PyTorch 1.8.1-era code and calls
    models.<backbone>(pretrained=True). Modern torchvision maps DEFAULT to newer
    recipes for some models (notably ResNet50 V2), so exact reproduction must
    request IMAGENET1K_V1 explicitly.
    """
    try:
        if name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            if pretrained:
                print("[Init] resnet18 legacy pretrained=True -> IMAGENET1K_V1")
            return models.resnet18(weights=weights)
        if name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            if pretrained:
                print("[Init] resnet50 legacy pretrained=True -> IMAGENET1K_V1")
            return models.resnet50(weights=weights)
        if name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            if pretrained:
                print("[Init] mobilenet_v2 legacy pretrained=True -> IMAGENET1K_V1")
            return models.mobilenet_v2(weights=weights)
    except AttributeError:
        # Old torchvision path: pretrained=True already means the historical V1
        # weights, which is exactly what the public CAST repository used.
        if name == "resnet18":
            return models.resnet18(pretrained=pretrained)
        if name == "resnet50":
            return models.resnet50(pretrained=pretrained)
        if name == "mobilenet_v2":
            return models.mobilenet_v2(pretrained=pretrained)
    raise ValueError("Backbone Error: %s" % name)


def cal_weight(x):
    # Public Networks.py behavior, intentionally unchanged.
    w = 1 / x
    w = np.exp(w) / np.sum(np.exp(w))
    return np.array(w)


def compute_kernel_matrix(source, target, kernel_mul=2.0, kernel_num=5):
    # Public Networks.py behavior: explicit pairwise tensor + SUM of kernels.
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    l2_distance = ((total0 - total1) ** 2).sum(2)
    bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_loss(source_features, target_features):
    # Public Networks.py behavior: random equal-size matching, repeated six times.
    num_samples = min(source_features.size(0), target_features.size(0))
    matched_source = source_features[torch.randperm(source_features.size(0))[:num_samples]]
    loss = 0.0
    for _ in range(6):
        matched_target = target_features[torch.randperm(target_features.size(0))[:num_samples]]
        kernels = compute_kernel_matrix(matched_source, matched_target)
        xx = kernels[:num_samples, :num_samples]
        yy = kernels[num_samples:, num_samples:]
        xy = kernels[:num_samples, num_samples:]
        yx = kernels[num_samples:, :num_samples]
        loss += torch.mean(xx + yy - xy - yx)
    return loss / 6


def remove_element(matrix, index):
    # Preserve the public implementation's row-flatten/cat behavior.
    dim = matrix[0].shape[-1]
    data = []
    for item in matrix[:index]:
        data.extend(item)
    for item in matrix[index + 1:]:
        data.extend(item)
    return torch.cat(data, 0).reshape(-1, dim)


class ExactCASTModel(nn.Module):
    """Inline copy of the public repository's Networks.Model behavior."""

    def __init__(self, backbone="resnet50", num_classes=7, pretrained=True, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(num_classes)
        self.kde = KernelDensity(bandwidth=0.2, kernel="gaussian")

        base = _build_backbone(backbone, pretrained)
        if backbone == "resnet18":
            self.feature = nn.Sequential(
                *list(base.children())[:-1], nn.Flatten(), nn.Dropout(drop_rate)
            )
            self.fc = nn.Linear(512, num_classes, bias=False)
        elif backbone == "resnet50":
            self.feature = nn.Sequential(
                *list(base.children())[:-1], nn.Flatten(), nn.Dropout(drop_rate), nn.Linear(2048, 512)
            )
            self.fc = nn.Linear(512, num_classes, bias=False)
        elif backbone == "mobilenet_v2":
            self.feature = nn.Sequential(
                *list(base.children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(drop_rate), nn.Linear(1280, 512), nn.Dropout(drop_rate)
            )
            self.fc = nn.Linear(512, num_classes, bias=False)
        else:
            raise ValueError("Backbone Error!")

    def split_feature_makeLD(self, x, target):
        x_parts = []
        for c in range(self.num_classes):
            ind = (target == c).nonzero()
            x_parts.append(x[ind[:, 0], :])
        return x_parts

    def volume(self, features):
        # Intentionally preserve public code: score_samples is log-density and
        # the repository directly takes its reciprocal before cal_weight.
        volume = np.zeros(shape=(len(features)))
        for idx in range(len(features)):
            f = features[idx].cpu().detach().numpy()
            if len(f) != 0:
                self.kde.fit(f)
                v = np.sum(1 / (self.kde.score_samples(f)))
            else:
                v = 0.000001
            volume[idx] = v
        return cal_weight(volume)

    def forward(self, x, targets, idx, mode="train", task="target", epoch=0):
        fea = self.feature(x)
        out = self.bn(self.fc(fea))
        batch = fea.shape[0]

        if mode == "train":
            if task == "source":
                features = self.split_feature_makeLD(fea, targets)
                weight = self.volume(features)
                w = torch.from_numpy(np.array(weight)).to(fea.device)
                inter_loss = torch.tensor(0.0, device=fea.device)
                for i in range(7):
                    fea_c = features[i]
                    if len(fea_c) != 0:
                        fea_other = remove_element(features, i).to(fea.device)
                        inter_loss += mmd_loss(fea_c, fea_other) * w[i]
                return [out, -1 * inter_loss]

            if task == "target":
                confident_idx = (idx == 1).nonzero().squeeze()
                fea = torch.index_select(fea, 0, confident_idx)
                targets = torch.index_select(targets, 0, confident_idx)

                # Public repository splits with the ORIGINAL concatenated batch/2.
                # This is intentionally retained, including last-batch behavior.
                source_features = self.split_feature_makeLD(fea[:batch // 2], targets[:batch // 2])
                target_features = self.split_feature_makeLD(fea[batch // 2:], targets[batch // 2:])
                features = self.split_feature_makeLD(fea, targets)
                weight = self.volume(features)
                w = torch.from_numpy(np.array(weight)).to(fea.device)
                intra_loss = torch.tensor(0.0, device=fea.device)
                inter_loss = torch.tensor(0.0, device=fea.device)

                for i in range(7):
                    fea_s = source_features[i]
                    fea_t = target_features[i]
                    if len(fea_s) != 0 and len(fea_t) != 0:
                        intra_loss += mmd_loss(fea_s, fea_t) * w[i]

                    fea_c = features[i]
                    if len(fea_c) != 0:
                        fea_other = remove_element(features, i).to(fea.device)
                        inter_loss += mmd_loss(fea_c, fea_other) * w[i]

                return [out, intra_loss - inter_loss]

        return out, fea.cpu()


class ExactRafDataSet(Dataset):
    """Public dataset.py RAF behavior with a configurable root."""

    def __init__(self, raf_path, phase, transform=None, strong_transform=None):
        self.phase = phase
        self.transform = transform
        self.strong_transform = strong_transform
        df = pd.read_csv(os.path.join(raf_path, "EmoLabel/list_patition_label.txt"), sep=" ", header=None)
        dataset = df[df[0].str.startswith("train" if phase == "train" else "test")]
        file_names = dataset.iloc[:, 0].values.copy()
        labels = dataset.iloc[:, 1].values.copy() - 1

        np.random.seed(2000)
        np.random.shuffle(file_names)
        np.random.seed(2000)
        np.random.shuffle(labels)
        self.label = [int(x) for x in labels]
        self.file_paths = []
        for f in file_names:
            aligned = f.split(".")[0] + "_aligned.jpg"
            self.file_paths.append(os.path.join(raf_path, "Image/aligned", aligned))
        self._print_distribution("RAF %s" % phase)

    def _print_distribution(self, name):
        arr = np.asarray(self.label)
        counts = [int(np.sum(arr == c)) for c in range(7)]
        print("[%s] images=%d CAST_counts=%s" % (name, len(self.label), counts))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.file_paths[idx])
        if image is None:
            raise FileNotFoundError(self.file_paths[idx])
        img = image[:, :, ::-1]
        label = self.label[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.strong_transform is not None:
            # Preserve public RAF dataset behavior: strong transform receives
            # the already weak-transformed tensor. Source strong view is unused.
            img_aug = self.strong_transform(img)
            return img, img_aug, label
        return img, label


def _resolve_split(root: str, split: str) -> str:
    candidates = {
        "train": ["train", "Train", "training"],
        "val": ["val", "valid", "validation", "publictest", "PublicTest"],
        "test": ["test", "Test", "private_test", "privatetest", "PrivateTest"],
    }[split]
    for name in candidates:
        p = Path(root) / name
        if p.is_dir():
            return str(p)
    raise FileNotFoundError("FER split %s not found under %s" % (split, root))


class ExactFERDataSet(Dataset):
    """Public FER behavior adapted to canonical FER2013 train/val/test folders.

    Repo-exact target training is one folder named 'train'. The paper Table-I
    counts reveal that this folder contains canonical train + public validation
    (28709 + 3589 = 32298). We merge those two physical splits in memory.
    """

    def __init__(self, root, phase, transform=None, strong_transform=None,
                 folder_order="kaggle", combine_val=True, eval_split="test"):
        self.phase = phase
        self.transform = transform
        self.strong_transform = strong_transform
        split_names = ["train"]
        if phase == "train" and combine_val:
            split_names.append("val")
        if phase != "train":
            split_names = [eval_split]

        files = []
        for split in split_names:
            split_dir = _resolve_split(root, split)
            files.extend(glob.glob(os.path.join(split_dir, "*", "*.jpg")))
            files.extend(glob.glob(os.path.join(split_dir, "*", "*.jpeg")))
            files.extend(glob.glob(os.path.join(split_dir, "*", "*.png")))

        if phase == "train":
            np.random.seed(2000)
            np.random.shuffle(files)

        self.file_paths = []
        self.label = []
        for file in files:
            raw = int(os.path.basename(os.path.dirname(file)))
            label = KAGGLE_TO_CAST[raw] if folder_order == "kaggle" else raw
            self.file_paths.append(file)
            self.label.append(int(label))

        arr = np.asarray(self.label)
        counts = [int(np.sum(arr == c)) for c in range(7)]
        print("[FER %s] physical_splits=%s images=%d CAST_counts=%s" % (
            phase, split_names, len(self.label), counts
        ))
        if phase == "train" and combine_val and folder_order == "kaggle":
            print("[FER exact check] expected paper Table-I counts=%s match=%s" % (
                PAPER_FER_TRAIN_COUNTS, counts == PAPER_FER_TRAIN_COUNTS
            ))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.file_paths[idx])
        if image is None:
            raise FileNotFoundError(self.file_paths[idx])
        image = image[:, :, ::-1]
        label = self.label[idx]
        img = self.transform(image) if self.transform is not None else image
        if self.strong_transform is not None:
            # Preserve public FER behavior: strong transform starts from raw RGB.
            img_aug = self.strong_transform(image)
            return img, img_aug, label
        return img, label


def build_transforms():
    # Exact public train.py transforms (not the v6 transforms).
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomCrop(224, padding=32),
        ], p=0.5),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
    test_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    augment_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomCrop(224, padding=32),
        ], p=0.5),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
    return train_tf, test_tf, augment_tf


def annotate_target(pred, class_num, epoch, total_epochs, phi):
    # Exact public train.py CATM rule: computed independently per mini-batch.
    prob = F.softmax(pred.cpu(), 1)
    pred_values, pred_targets = torch.max(prob, dim=1)
    max_index = F.one_hot(pred_targets, class_num)
    preds_mean = np.transpose((prob * max_index).detach().numpy())
    class_sum = [np.sum(x) for x in preds_mean]
    class_idx = [len(np.where(x > 0)[0]) for x in preds_mean]
    class_mean = np.array([
        class_sum[i] / class_idx[i] if class_idx[i] != 0 else 0
        for i in range(len(class_idx))
    ])
    class_mean = np.array([
        mean * phi * (total_epochs / (total_epochs - epoch))
        for mean in class_mean
    ])
    class_mean = torch.from_numpy(np.minimum(class_mean, 0.9))
    threshold = class_mean.numpy()
    batch_mean = torch.index_select(class_mean, 0, pred_targets).detach().numpy()
    confident_ids = torch.from_numpy((pred_values.detach().numpy() > batch_mean).nonzero()[0])

    selected_pred = pred_targets.index_select(0, confident_ids).numpy()
    label_dis = [int(np.sum(selected_pred == c)) for c in range(class_num)]
    ones = torch.ones(confident_ids.shape[0])
    confident_mask = torch.zeros(prob.shape[0]).index_put([torch.LongTensor(confident_ids)], ones)
    return pred_targets, confident_mask, threshold, label_dis


def classifier_modulation_loss(model, device):
    # Exact public train.py CSCM expression, including diagonal constants.
    fc_weight = model.fc.weight
    fc_weight_norm = torch.norm(fc_weight, dim=1).unsqueeze(1)
    fc_weight_dot = fc_weight.mm(torch.transpose(fc_weight, 1, 0))
    fc_weight_norm_outer = fc_weight_norm.mm(torch.transpose(fc_weight_norm, 1, 0))
    return torch.mean(((fc_weight_dot / fc_weight_norm_outer - torch.eye(fc_weight.shape[0], device=device)) + 1) / 2)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    n = 0
    correct = 0
    class_correct = torch.zeros(7, dtype=torch.long)
    class_total = torch.zeros(7, dtype=torch.long)
    predicted = torch.zeros(7, dtype=torch.long)
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        out, _ = model(imgs, targets, None, mode="test")
        loss_sum += float(criterion(out, targets).sum().item())
        pred = out.argmax(dim=1)
        n += int(targets.numel())
        correct += int((pred == targets).sum().item())
        for c in range(7):
            mask = targets == c
            class_total[c] += int(mask.sum().item())
            class_correct[c] += int(((pred == targets) & mask).sum().item())
            predicted[c] += int((pred == c).sum().item())
    return {
        "acc": correct / max(1, n),
        "loss": loss_sum / max(1, n),
        "class_acc": (class_correct.float() / class_total.clamp_min(1).float()).tolist(),
        "class_total": class_total.tolist(),
        "predicted": predicted.tolist(),
    }


def save_best(model, optimizer, path, epoch, acc, stage):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "acc": acc,
        "stage": stage,
    }, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The public CAST implementation assumes CUDA; no CUDA device is available.")
    device = torch.device("cuda")

    if args.selection_split == "val" and not args.no_combine_target_val:
        raise ValueError("--selection-split val requires --no-combine-target-val to avoid using val in target training")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not args.model_dir:
        args.model_dir = os.path.join("models", "cast_exact_rafdb_fer", stamp)
    os.makedirs(args.model_dir, exist_ok=True)
    best_path = os.path.join(args.model_dir, "best_repo_protocol.pth")
    log_path = os.path.join(args.model_dir, "metrics.jsonl")

    print("CAST exact public-repo configuration:", json.dumps(vars(args), sort_keys=True))
    print("[Protocol] Public-repo exact mode uses target %s for checkpoint selection." % args.selection_split)
    if args.selection_split == "test":
        print("[Protocol][WARN] This intentionally reproduces target-test model-selection leakage from the public repo.")
    print("[FER label map] canonical FER2013/Kaggle -> CAST: 0:angry->5, 1:disgust->2, 2:fear->1, 3:happy->3, 4:sad->4, 5:surprise->0, 6:neutral->6")

    train_tf, test_tf, augment_tf = build_transforms()
    combine_val = not args.no_combine_target_val

    source_train = ExactRafDataSet(
        args.source_root, phase="train", transform=train_tf, strong_transform=augment_tf
    )
    source_test = ExactRafDataSet(
        args.source_root, phase="test", transform=test_tf, strong_transform=None
    )
    target_train = ExactFERDataSet(
        args.target_root, phase="train", transform=train_tf, strong_transform=augment_tf,
        folder_order=args.fer_folder_order, combine_val=combine_val
    )
    target_select = ExactFERDataSet(
        args.target_root, phase="eval", transform=test_tf, strong_transform=None,
        folder_order=args.fer_folder_order, combine_val=False, eval_split=args.selection_split
    )
    target_test = target_select if args.selection_split == "test" else ExactFERDataSet(
        args.target_root, phase="eval", transform=test_tf, strong_transform=None,
        folder_order=args.fer_folder_order, combine_val=False, eval_split="test"
    )

    if args.backbone == "resnet50":
        train_batch, test_batch = 128, 100
    else:
        train_batch, test_batch = 128, 128

    source_train_loader = DataLoader(
        source_train, batch_size=train_batch, num_workers=args.workers,
        shuffle=True, pin_memory=True
    )
    source_test_loader = DataLoader(
        source_test, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True
    )
    target_train_loader = DataLoader(
        target_train, batch_size=train_batch, num_workers=args.workers,
        shuffle=True, pin_memory=True
    )
    target_select_loader = DataLoader(
        target_select, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True
    )
    target_test_loader = target_select_loader if args.selection_split == "test" else DataLoader(
        target_test, batch_size=test_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True
    )

    model = ExactCASTModel(
        backbone=args.backbone,
        num_classes=7,
        pretrained=not args.no_imagenet_pretrained,
    )
    if args.checkpoint:
        print("Loading checkpoint:", args.checkpoint)
        checkpoint = load_checkpoint(args.checkpoint, device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state, strict=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    best_acc = 0.0

    # ------------------------------------------------------------------
    # Phase 1: exact public-repo source training.
    # Public repo evaluates TARGET set every source epoch and selects on it.
    # ------------------------------------------------------------------
    for epoch in range(args.source_epochs):
        model.train()
        src_correct = 0
        src_seen = 0
        cls_sum = 0.0
        aff_sum = 0.0
        cscm_sum = 0.0
        batches = 0

        for imgs, _, targets in source_train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(imgs, targets, None, "train", "source")
            cls_loss = torch.mean(criterion(output[0], targets))
            weight_loss = classifier_modulation_loss(model, device)
            aff_loss = output[1]
            loss = cls_loss * args.w1 + aff_loss * args.w2 + weight_loss * args.w3

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pred = output[0].argmax(dim=1)
            src_correct += int((pred == targets).sum().item())
            src_seen += int(targets.numel())
            cls_sum += float(cls_loss.detach().item())
            aff_sum += float(aff_loss.detach().item())
            cscm_sum += float(weight_loss.detach().item())
            batches += 1

        scheduler.step()
        select_metrics = evaluate(model, target_select_loader, criterion, device)
        improved = select_metrics["acc"] > best_acc
        if improved:
            best_acc = select_metrics["acc"]
            save_best(model, optimizer, best_path, epoch, best_acc, "source")

        rec = {
            "stage": "source", "epoch": epoch,
            "source_train_acc": src_correct / max(1, src_seen),
            "cls": cls_sum / max(1, batches),
            "aff": aff_sum / max(1, batches),
            "cscm": cscm_sum / max(1, batches),
            "lr": optimizer.param_groups[0]["lr"],
            "target_selection": select_metrics,
            "best_acc": best_acc,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("[Source %02d] train_acc=%.4f cls=%.4f aff=%.4f cscm=%.4f lr=%.6f target_%s=%.4f best=%.4f" % (
            epoch, rec["source_train_acc"], rec["cls"], rec["aff"], rec["cscm"],
            rec["lr"], args.selection_split, select_metrics["acc"], best_acc
        ))

    if not os.path.exists(best_path):
        raise RuntimeError("No source checkpoint was saved")
    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("[Source -> Target] loaded repo-selected source checkpoint: stage=%s epoch=%s acc=%.4f" % (
        checkpoint.get("stage"), checkpoint.get("epoch"), checkpoint.get("acc", -1.0)
    ))
    source_eval = evaluate(model, source_test_loader, criterion, device)
    target_start = evaluate(model, target_test_loader, criterion, device)
    print("[Source][RAF test] acc=%.4f class_acc=%s" % (
        source_eval["acc"], [round(x, 4) for x in source_eval["class_acc"]]
    ))
    print("[Source -> Target test] acc=%.4f class_acc=%s predicted=%s" % (
        target_start["acc"], [round(x, 4) for x in target_start["class_acc"]], target_start["predicted"]
    ))

    # ------------------------------------------------------------------
    # Phase 2: exact public-repo target self-training.
    # Keep the source iterator lazy, as in public train.py: it is first created
    # only after the first target batch has already been drawn.
    # ------------------------------------------------------------------
    source_train_iter = None
    for epoch in range(args.epochs):
        cls_sum = 0.0
        aff_sum = 0.0
        batches = 0
        confident_num = 0
        pseudo_correct = 0
        pseudo_selected = 0
        threshold_sum = np.zeros(7, dtype=np.float64)
        threshold_batches = 0
        selected_dist = np.zeros(7, dtype=np.int64)

        for imgs, imgs_aug, gt_target in target_train_loader:
            try:
                if source_train_iter is None:
                    raise StopIteration
                source_imgs, _, source_targets = next(source_train_iter)
            except StopIteration:
                source_train_iter = iter(source_train_loader)
                source_imgs, _, source_targets = next(source_train_iter)

            model.eval()
            with torch.no_grad():
                out, _ = model(imgs.to(device, non_blocking=True), None, None, "test", "target")
            targets, con_idx, threshold, label_dis = annotate_target(
                out, 7, epoch, args.epochs, args.phi
            )

            # Diagnostics only. gt_target never influences optimization.
            chosen = con_idx.bool()
            if int(chosen.sum().item()) > 0:
                pseudo_correct += int((targets[chosen] == gt_target[chosen]).sum().item())
                pseudo_selected += int(chosen.sum().item())
            threshold_sum += threshold
            threshold_batches += 1
            selected_dist += np.asarray(label_dis, dtype=np.int64)
            confident_num += int(chosen.sum().item())

            model.train()
            source_con_idx = torch.ones(source_imgs.shape[0])
            train_imgs = torch.cat((source_imgs, imgs_aug), 0).to(device, non_blocking=True)
            train_targets = torch.cat((source_targets, targets), 0).to(device, non_blocking=True)
            train_con_idx = torch.cat((source_con_idx, con_idx), 0).to(device, non_blocking=True)

            output = model(train_imgs, train_targets, train_con_idx, "train", "target")
            cls_loss = torch.mean(criterion(output[0], train_targets) * train_con_idx)
            weight_loss = classifier_modulation_loss(model, device)
            aff_loss = output[1]
            loss = cls_loss * args.w1 + aff_loss * args.w2 + weight_loss * args.w3

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cls_sum += float(cls_loss.detach().item())
            aff_sum += float(aff_loss.detach().item())
            batches += 1

        scheduler.step()
        select_metrics = evaluate(model, target_select_loader, criterion, device)
        improved = select_metrics["acc"] > best_acc
        if improved:
            best_acc = select_metrics["acc"]
            save_best(model, optimizer, best_path, epoch, best_acc, "target")

        avg_threshold = (threshold_sum / max(1, threshold_batches)).tolist()
        pseudo_acc = pseudo_correct / max(1, pseudo_selected)
        rec = {
            "stage": "target", "epoch": epoch,
            "confident_num": confident_num,
            "selected_dist": selected_dist.tolist(),
            "pseudo_acc_debug": pseudo_acc,
            "avg_threshold": avg_threshold,
            "cls": cls_sum / max(1, batches),
            "aff": aff_sum / max(1, batches),
            "lr": optimizer.param_groups[0]["lr"],
            "target_selection": select_metrics,
            "best_acc": best_acc,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("[Target %02d] confident=%d dist=%s pseudo_acc=%.4f thr=%s cls=%.4f aff=%.4f lr=%.6f target_%s=%.4f best=%.4f" % (
            epoch, confident_num, selected_dist.tolist(), pseudo_acc,
            [round(x, 4) for x in avg_threshold], rec["cls"], rec["aff"], rec["lr"],
            args.selection_split, select_metrics["acc"], best_acc
        ))

    best = load_checkpoint(best_path, device)
    model.load_state_dict(best["model"], strict=True)
    final_test = evaluate(model, target_test_loader, criterion, device)
    print("[FINAL] best_stage=%s best_epoch=%s selection_acc=%.4f target_test_acc=%.4f" % (
        best.get("stage"), best.get("epoch"), best.get("acc", -1.0), final_test["acc"]
    ))
    print("[FINAL] target_test_class_acc=%s predicted=%s" % (
        [round(x, 4) for x in final_test["class_acc"]], final_test["predicted"]
    ))
    print("Logs:", log_path)
    print("Checkpoint:", best_path)


if __name__ == "__main__":
    main()
