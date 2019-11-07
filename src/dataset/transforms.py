__author__ = "charles"
__email__ = "charleschen2013@163.com"

import collections
import random

import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms

import torch
import math
import numbers

KEYS = ('image', 'instance', 'label')
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MEAN_0_5 = (0.5, 0.5, 0.5)
STD_0_5 = (0.5, 0.5, 0.5)


def get_transform(args):
    """
    1. 'object_center',  'crop', 'resize_crop'
    2. 'color'
    3. default Hflip
    4. 'bbox'
    5. 'std'
    :param args:
    :return:
    """
    aug = args.aug
    transform_list = []

    # crop ways
    if 'resize' in aug:
        transform_list.append(transforms.Resize(args.crop_size))
    elif 'crop' in aug:
        transform_list.append(transforms.RandomCrop(args.crop_size, padding=4))
        pass
    # HFlip:
    if 'hflip' in aug:
        transform_list.append(transforms.RandomHorizontalFlip())

    # ToTensor
    transform_list.append(transforms.ToTensor())

    transform_list.append(transforms.Normalize(mean=MEAN_0_5, std=STD_0_5))

    # Train transform
    print('===> Transform:')
    for i, tf in enumerate(transform_list):
        print(i + 1, tf.__class__.__name__)
    print('==========\n')
    train_transform = transforms.Compose(transform_list)

    # Val transform
    val_transform_list = []
    val_transform_list.append(transforms.ToTensor())
    val_transform_list.append(transforms.Normalize(mean=MEAN_0_5, std=STD_0_5))

    print('===> Transform:')
    for i, tf in enumerate(val_transform_list):
        print(i + 1, tf.__class__.__name__)
    print('==========\n')
    val_transform = transforms.Compose(val_transform_list)

    return train_transform, val_transform


if __name__ == '__main__':
    from utils.train_config import config

    args = config()
    train_transform, val_transform = get_transform(args)
    print(train_transform, val_transform)
