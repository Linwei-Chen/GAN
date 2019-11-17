from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from src.style_transfer_perceptual_loss.train_config import config
import random
import collections
import random

import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms as T

import torch
import math
import numbers

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_image_dataset(args, train=True):
    train_transform, _ = get_transform(args)
    if train:
        train_dataset = datasets.ImageFolder(args.image_dataset, train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.prefetch, pin_memory=False, drop_last=True)
        return train_loader
    else:
        train_dataset = datasets.ImageFolder(args.image_dataset, transforms.Compose([
            StrideAlign(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]))
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                  num_workers=args.prefetch, pin_memory=False, drop_last=True)
        return train_loader


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
        transform_list.append(transforms.RandomCrop(args.crop_size, pad_if_needed=True))
        pass
    # HFlip:
    if 'hflip' in aug:
        transform_list.append(transforms.RandomHorizontalFlip())

    # ToTensor
    transform_list.append(transforms.ToTensor())

    transform_list.append(transforms.Normalize(mean=MEAN, std=STD))

    # Train transform
    print('===> Transform:')
    for i, tf in enumerate(transform_list):
        print(i + 1, tf.__class__.__name__)
    print('==========\n')
    train_transform = transforms.Compose(transform_list)

    # Val transform
    val_transform_list = [transforms.Resize(args.crop_size)]
    val_transform_list.append(transforms.ToTensor())
    val_transform_list.append(transforms.Normalize(mean=MEAN, std=STD))

    print('===> Transform:')
    for i, tf in enumerate(val_transform_list):
        print(i + 1, tf.__class__.__name__)
    print('==========\n')
    val_transform = transforms.Compose(val_transform_list)

    return train_transform, val_transform


class StrideAlign(object):
    def __init__(self, stride=16, resize_both=False, down_sample=None, interpolation=Image.BILINEAR):
        self.stride = stride
        self.interpolation = interpolation
        self.down_sample = down_sample
        self.resize_both = resize_both

    def __call__(self, img, target=None):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        w, h = img.size
        if self.down_sample is not None:
            w, h = w / self.down_sample, h / self.down_sample
        w_stride, h_stride = math.floor(w / self.stride), math.floor(h / self.stride)
        h_w_resize = (int(h_stride * self.stride), int(w_stride * self.stride))
        img = F.resize(img, h_w_resize, self.interpolation)
        if self.resize_both:
            target = F.resize(target, h_w_resize, Image.NEAREST)
        # img_resized.show()
        return img

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


if __name__ == '__main__':
    args = config()
    data_loader = get_image_dataset(args)
    for s in data_loader:
        print(s[0].shape)
