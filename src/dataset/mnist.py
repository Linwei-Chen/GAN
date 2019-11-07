__author__ = "charles"
__email__ = "charleschen2013@163.com"

import argparse
import os
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import time
from utils.logger import ModelSaver, Logger
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from PIL import Image


class ToRGB:
    def __init__(self):
        pass

    def __call__(self, image):
        return image.convert("RGB")


def get_mnist_data_set(args, train=True):
    from dataset.transforms import get_transform

    # Init dataset
    if not os.path.isdir(args.cifar_10_data_path):
        os.makedirs(args.cifar_10_data_path)

    train_transform, val_transform = get_transform(args)
    train_transform = transforms.Compose([ToRGB(), train_transform])
    val_transform = transforms.Compose([ToRGB(), val_transform])

    if train:
        train_data = dset.MNIST(args.mnist_data_path, train=True, download=True, transform=train_transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.prefetch, pin_memory=False, drop_last=True)
        return train_loader
    else:
        test_data = dset.MNIST(args.mnist_data_path, train=False, download=True, transform=train_transform)

        test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.prefetch, pin_memory=False)
        return test_loader
