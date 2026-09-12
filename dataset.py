import torch.utils.data as data
import cv2
import pandas as pd
import os
import image_utils as util
import random
import glob
import numpy as np
import random
import shutil



"0: surprise, 1: fear, 2: disgust, 3: happy 4: sad  5: angry 6: neutral 7:attempt"

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, strong_transform =None, basic_aug=False, ratio=1):
        self.phase = phase
        self.transform = transform
        self.strong_transform = strong_transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:,
                     LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        ###shuffle dataset
        seed = np.random.seed(2000)
        np.random.shuffle(file_names)
        seed = np.random.seed(2000)
        np.random.shuffle(self.label)
                
        self.file_paths = []
        # use raf-db aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        distribute = np.array(self.label)

        self.label_dis = [np.sum(distribute == 0), np.sum(distribute == 1), np.sum(distribute == 2),
                          np.sum(distribute == 3), \
                          np.sum(distribute == 4), np.sum(distribute == 5), np.sum(distribute == 6)]
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (
        self.label_dis[0], self.label_dis[1], self.label_dis[2], self.label_dis[3], \
        self.label_dis[4], self.label_dis[5], self.label_dis[6]))

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape=len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        img = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 2)
                img = self.aug_func[index](img)

        if self.transform is not None:
            img = self.transform(img)
            
        if self.strong_transform is not None:
            img_aug = self.strong_transform(img)
            return img, img_aug, label
        else:
            return img, label


class FER(data.Dataset):
    def __init__(self, path, phase, transform=None, strong_transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.strong_transform = strong_transform

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        self.file_paths, self.label = [], []

        if self.phase == 'train':
            files = glob.glob(
                os.path.join(
                    path,
                    'train',
                    '*',
                    '*.jpg'
                )
            )
            seed = np.random.seed(2000)
            np.random.shuffle(files)
        else:
            files = glob.glob(os.path.join(path, 'test/*/*.jpg'))
            print("FER test images:", len(files))

        print("FER %s path: %s" % (
                    self.phase,os.path.abspath(path) ) )

        print(
                "FER %s found images: %d" % (
                    self.phase,len(files)) )

            # FER2013 -> CAST label mapping
        fer_to_cast = {
            0: 5,  # angry
            1: 2,  # disgust
            2: 1,  # fear
            3: 3,  # happy
            4: 4,  # sad
            5: 0,  # surprise
            6: 6,  # neutral
            }

        for file in files:
            self.file_paths.append(file)

            original_label = int(
                os.path.basename(os.path.dirname(file))
            )
            mapped_label = fer_to_cast[original_label]

            self.label.append(mapped_label)


        distribute = np.array(self.label)

        self.label_dis = [
            np.sum(distribute == 0),
            np.sum(distribute == 1),
            np.sum(distribute == 2),
            np.sum(distribute == 3),
            np.sum(distribute == 4),
            np.sum(distribute == 5),
            np.sum(distribute == 6)
        ]

        print(
            'The dataset distribute: %d, %d, %d, %d, %d, %d, %d'
            % (
                self.label_dis[0],
                self.label_dis[1],
                self.label_dis[2],
                self.label_dis[3],
                self.label_dis[4],
                self.label_dis[5],
                self.label_dis[6]
            )
        )


    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape = len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        image = cv2.imread(path)

        if image is None:
            raise FileNotFoundError(
                "Failed to read image: %s" % path
            )

        # BGR -> RGB
        image = image[:, :, ::-1]

        label = self.label[idx]

        # --------------------------------------------------
        # Basic augmentation
        # --------------------------------------------------
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        # ==================================================
        # E3: Two independent target views
        # ==================================================
        if self.transform is not None:

            # View 1
            img_w1 = self.transform(image)

            # View 2
            # Calling the same random transform again gives
            # another independent augmentation of the same image.
            if self.phase == 'train':
                img_w2 = self.transform(image)
            else:
                img_w2 = None

        else:
            img_w1 = image
            img_w2 = image if self.phase == 'train' else None

        # ==================================================
        # Target training:
        # two Teacher views + one strong Student view
        # ==================================================
        if self.strong_transform is not None:

            img_aug = self.strong_transform(image)

            return (
                img_w1,
                img_w2,
                img_aug,
                label
            )


        # Validation / test
        else:
            return img_w1, label
if __name__ == '__main__':
    train_dataset = FER(
        '/workspace/ttt/code/data/fer2013/',
        phase='train',
        transform=None,
        strong_transform=None,
        basic_aug=False
    )

    test_dataset = FER(
        '/workspace/ttt/code/data/fer2013/',
        phase='test',
        transform=None,
        strong_transform=None,
        basic_aug=False
    )

    print('train size:', len(train_dataset))
    print('test size:', len(test_dataset))

    print('train distribution:', train_dataset.label_dis)
    print('test distribution:', test_dataset.label_dis)

