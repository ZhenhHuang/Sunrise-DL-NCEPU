import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import cv2
from utils.preprocess import flip, noise, rotation


def unzip(src, dst):
    if zipfile.is_zipfile(src):
        f = zipfile.ZipFile(src, 'r')
        for file in f.namelist():
            f.extract(file, dst)
    else:
        raise FileNotFoundError


def normalize(img):
    # H, W, C
    # mean = img.mean()
    # std = img.std()
    return img / 255.


def image_augment(img):
    opt = np.random.uniform(0, 1)

    def get_method(opt):
        if opt < 0.25:
            return flip
        elif opt < 0.5:
            return rotation
        elif opt < 0.75:
            return noise
        else:
            return None

    method = get_method(opt)
    return method(img) if method is not None else img


note = r"C:/Users/98311/PycharmProjects/PaddleProjects/data/data146107"


class Caltech101(Dataset):
    def __init__(self, root_path=note, data_path='dataset', flag='train', size=None, abandon=True, factor=2):
        super(Caltech101, self).__init__()
        assert flag in ['train', 'val', 'test', 'pred']
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.abandon = abandon
        self.factor = factor
        if flag == 'pred' and abandon:
            self.abandon = False
        if size is None:
            self.height = 200
            self.width = 200
        else:
            self.height = size[0]
            self.width = size[1]
        self.__read_data__()

    def __getitem__(self, index):
        if self.flag == 'pred':
            return self.data[index], self.data_names[index]
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def __read_data__(self):
        src = f"{self.root_path}/dataset.zip"
        dst = self.root_path

        if not os.path.exists(f"{self.root_path}/{self.data_path}"):
            unzip(src, dst)

        path = os.path.join(self.root_path, self.data_path)
        self.map_dict = self.__getClassMap(path)
        target_path = os.path.join(path, "test.txt") if self.flag == 'pred' else os.path.join(path, "train.txt")

        f = open(target_path)
        data = []
        label = []
        data_names = []
        lines = f.readlines()

        if self.flag != "pred":
            mapping = {'train': 0, 'val': 1, 'test': 2}
            n = len(lines)
            train = int(n * 0.8)
            val = int(n * 0.1)
            border1 = [0, train, train + val]
            border2 = [train, train + val, n]
            index1 = border1[mapping[self.flag]]
            index2 = border2[mapping[self.flag]]
            lines = lines[index1: index2]

        for line in lines:
            if self.flag != 'pred':
                pic_name, cls = line.split()
            else:
                pic_name = line.strip()
            data_names.append(pic_name)
            img = cv2.imread(f'{path}/images/{pic_name}')
            # img = self.__preprocess(img, self.abandon, self.factor)
            # if self.flag != 'pred':
            #     img = image_augment(img)
            img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_LINEAR)
            if img is not None:
                if self.flag != 'pred':
                    label.append(int(cls))
                data.append(normalize(img))
        f.close()

        self.data = np.array(data)
        self.label = np.array(label)
        self.data_names = data_names

    def __getCountDict(self):
        self.count_dict = {}
        for i in self.label:
            self.count_dict[i] = self.count_dict.get(i, 0) + 1

    def __getClassMap(self, path):
        f = open(f"{path}/class.txt")
        map_dict = {}
        for line in f.readlines():
            name, id = line.split()
            map_dict[name] = int(id)
        f.close()
        return map_dict

    def __preprocess(self, img, abandon=True, factor=2, resize_only=True):
        H, W, C = img.shape
        if resize_only:
            img = img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_LINEAR)
            return img
        if H > factor * self.height or W > factor * self.width or H < self.height // factor or W < self.width // factor:
            if abandon:
                return None
            else:
                img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_LINEAR)
                return img

        if H < self.height:
            pad = self.height - H
            img = cv2.copyMakeBorder(img, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if W < self.width:
            pad = self.width - W
            img = cv2.copyMakeBorder(img, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if H > self.height:
            crop = H - self.height
            img = img[crop // 2:crop // 2 - crop, :, :]
        if W > self.width:
            crop = W - self.width
            img = img[:, crop // 2:crop // 2 - crop, :]
        return img


def getLoader(args, flag):
    if flag == 'train':
        batch_size = args.batch_size
        shuffle = True
        drop_last = True
    elif flag == 'pred':
        batch_size = 1
        shuffle = False
        drop_last = False
    else:
        batch_size = 1
        shuffle = False
        drop_last = True
    dataset = Caltech101(args.root_path, args.data_path, flag, args.size, args.abandon, args.factor)
    print(f"{flag}: {len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataset, data_loader

