import cv2
import numpy as np
import albumentations as A
from torchvision.transforms import RandomHorizontalFlip


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item):
        item_ = item.copy()
        for t in self.transforms:
            item_ = t(item_)
        return item_


class Resize(object):
    def __init__(self, size=(320, 32)):
        self.size = size

    def __call__(self, item):
        item['image'] = cv2.resize(item['image'], self.size)
        return item


class ImgRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.flip = RandomHorizontalFlip(p)

    def __call__(self, item):
        if np.random.binomial(1, self.p, 1)[0] == 1:
            item["image"] = cv2.flip(item["image"], 1)
        # item["image"] = self.flip(item["image"])

        return item


class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)):
        self.mean = np.asarray(mean).reshape((1, 1, 3)).astype(np.float32)
        self.std = np.asarray(std).reshape((1, 1, 3)).astype(np.float32)

    def __call__(self, item):
        item["image"] = (item["image"] - self.mean) / self.std
        return item


def get_train_transforms(image_size):
    return Compose([
        # Normalize(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # A.augmentations.transforms.HorizontalFlip(),
        ImgRandomHorizontalFlip(0.5),
        Resize(size=image_size),
    ])
