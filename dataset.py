import os
import glob
import random

import cv2
import numpy as np
import pandas as pd
import torch.utils.data as data

import image_utils as util


# CAST label order:
# 0: surprise, 1: fear, 2: disgust, 3: happy, 4: sad, 5: angry, 6: neutral


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, strong_transform=None, basic_aug=False, ratio=1):
        self.phase = phase
        self.transform = transform
        self.strong_transform = strong_transform
        self.raf_path = raf_path

        name_column = 0
        label_column = 1
        df = pd.read_csv(
            os.path.join(self.raf_path, "EmoLabel/list_patition_label.txt"),
            sep=" ",
            header=None,
        )

        if phase == "train":
            dataset = df[df[name_column].str.startswith("train")]
        else:
            dataset = df[df[name_column].str.startswith("test")]

        file_names = dataset.iloc[:, name_column].to_numpy(copy=True)
        labels = dataset.iloc[:, label_column].to_numpy(copy=True) - 1

        # Deterministic paired shuffle. Using one permutation is safer than
        # independently shuffling filenames and labels.
        rng = np.random.RandomState(2000)
        perm = rng.permutation(len(file_names))
        file_names = file_names[perm]
        self.label = labels[perm]

        self.file_paths = []
        for f in file_names:
            stem = f.split(".")[0]
            path = os.path.join(self.raf_path, "Image/aligned", stem + "_aligned.jpg")
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [
            util.flip_image,
            util.add_gaussian_noise,
            util.crop,
            util.rotation,
        ]

        distribute = np.asarray(self.label)
        self.label_dis = [int(np.sum(distribute == c)) for c in range(7)]
        print(
            "The dataset distribute: %d, %d, %d, %d, %d, %d, %d"
            % tuple(self.label_dis)
        )

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        counts = np.asarray(self.label_dis, dtype=np.float32)
        return np.ones_like(counts) / np.maximum(counts, 1.0) * 1000.0

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError("Failed to read image: %s" % path)

        image = image[:, :, ::-1]  # BGR -> RGB
        label = int(self.label[idx])

        if self.phase == "train" and self.basic_aug and random.uniform(0, 1) > 0.5:
            aug_idx = random.randint(0, 2)
            image = self.aug_func[aug_idx](image)

        img = self.transform(image) if self.transform is not None else image

        # Important: strong_transform must receive the raw RGB image, not an
        # already-normalized tensor.
        if self.strong_transform is not None:
            img_aug = self.strong_transform(image)
            return img, img_aug, label

        return img, label


class FER(data.Dataset):
    """FER2013 dataset with dual weak teacher views + one strong student view.

    Training return format when strong_transform is enabled:
        weak_view_1, weak_view_2, strong_view, mapped_label, sample_index

    The mapped label is returned only for diagnostics in UDA training. The
    training loss must not use it as target supervision.
    """

    def __init__(self, path, phase, transform=None, strong_transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.strong_transform = strong_transform
        self.basic_aug = basic_aug
        self.aug_func = [
            util.flip_image,
            util.add_gaussian_noise,
            util.crop,
            util.rotation,
        ]

        if self.phase == "train":
            files = sorted(glob.glob(os.path.join(path, "train", "*", "*.jpg")))
            rng = np.random.RandomState(2000)
            rng.shuffle(files)
        else:
            files = sorted(glob.glob(os.path.join(path, "test", "*", "*.jpg")))
            print("FER test images:", len(files))

        print("FER %s path: %s" % (self.phase, os.path.abspath(path)))
        print("FER %s found images: %d" % (self.phase, len(files)))

        # FER2013 -> CAST label mapping
        # FER: 0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral
        # CAST: 0 surprise, 1 fear, 2 disgust, 3 happy, 4 sad, 5 angry, 6 neutral
        fer_to_cast = {
            0: 5,
            1: 2,
            2: 1,
            3: 3,
            4: 4,
            5: 0,
            6: 6,
        }

        self.file_paths = []
        self.label = []
        for file in files:
            self.file_paths.append(file)
            original_label = int(os.path.basename(os.path.dirname(file)))
            self.label.append(fer_to_cast[original_label])

        distribute = np.asarray(self.label)
        self.label_dis = [int(np.sum(distribute == c)) for c in range(7)]
        print(
            "The dataset distribute: %d, %d, %d, %d, %d, %d, %d"
            % tuple(self.label_dis)
        )

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        counts = np.asarray(self.label_dis, dtype=np.float32)
        return np.ones_like(counts) / np.maximum(counts, 1.0) * 1000.0

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError("Failed to read image: %s" % path)

        image = image[:, :, ::-1]  # BGR -> RGB
        label = int(self.label[idx])

        if self.phase == "train" and self.basic_aug and random.uniform(0, 1) > 0.5:
            aug_idx = random.randint(0, 1)
            image = self.aug_func[aug_idx](image)

        if self.transform is not None:
            img_w1 = self.transform(image)
            img_w2 = self.transform(image) if self.phase == "train" else None
        else:
            img_w1 = image
            img_w2 = image if self.phase == "train" else None

        if self.phase == "train" and self.strong_transform is not None:
            img_aug = self.strong_transform(image)
            # sample index is needed by the epoch-level pseudo-label bank.
            return img_w1, img_w2, img_aug, label, idx

        # Validation / test remains compatible with the original 2-tuple API.
        return img_w1, label


if __name__ == "__main__":
    train_dataset = FER(
        "/workspace/ttt/code/data/fer2013/",
        phase="train",
        transform=None,
        strong_transform=None,
        basic_aug=False,
    )
    test_dataset = FER(
        "/workspace/ttt/code/data/fer2013/",
        phase="test",
        transform=None,
        strong_transform=None,
        basic_aug=False,
    )

    print("train size:", len(train_dataset))
    print("test size:", len(test_dataset))
    print("train distribution:", train_dataset.label_dis)
    print("test distribution:", test_dataset.label_dis)
