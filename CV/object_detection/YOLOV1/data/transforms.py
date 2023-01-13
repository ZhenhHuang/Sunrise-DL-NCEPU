import torchvision.transforms.functional as F
import torch
import random


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = F.hflip(image)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = F.vflip(image)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        _, H, W = image.shape
        image = F.resize(image, size=self.size)
        bbox = target["boxes"]
        bbox[:, [1, 3]] *= self.size[0] / H
        bbox[:, [0, 2]] *= self.size[1] / W
        target['boxes'] = bbox
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, self.mean, self.std)
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transfoms = transforms

    def __call__(self, image, target):
        for t in self.transfoms:
            image, target = t(image, target)
        return image, target

