__author__ = "charles"
__email__ = "charleschen2013@163.com"

import os
import torchvision.datasets as dset
from torch.utils.data import DataLoader


def get_cifar_10_data_set(args, train=True):
    from datasets.cls_dataset.transforms import get_transform

    # Init dataset
    if not os.path.isdir(args.cifar_10_data_path):
        os.makedirs(args.cifar_10_data_path)

    train_transform, val_transform = get_transform(args)
    if train:
        train_data = dset.CIFAR10(args.cifar_10_data_path, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.prefetch, pin_memory=False, drop_last=True)
        return train_loader
    else:
        test_data = dset.CIFAR10(args.cifar_10_data_path, train=False, transform=val_transform, download=True)

        test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.prefetch, pin_memory=False)
        return test_loader
