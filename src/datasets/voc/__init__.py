from datasets.voc.voc_instance_segmentation_dataset import VOCInstanceDataset
from datasets.transforms import get_transform
from torch.utils.data import DataLoader, ConcatDataset
from datasets.sbd import SBDInstanceDataset


def get_voc_instance_dataset(args, train, repeat):
    train_transform, val_transform = get_transform(args)

    if train:
        data_set = VOCInstanceDataset(data_dir=args.voc2012_data_path, transform=train_transform, train=True,
                                      choose_size=args.data_choose_size, repeat=repeat)
    else:
        data_set = VOCInstanceDataset(data_dir=args.voc2012_data_path, transform=val_transform, train=False)

    return data_set


def get_voc_instance_dataloader(args, train=True, repeat=1):
    assert 'balance' not in args.aug
    data_set = get_voc_instance_dataset(args, train, repeat)
    if train:
        data_loader = DataLoader(dataset=data_set,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.prefetch,
                                 pin_memory=False,
                                 drop_last=True)
    else:
        data_loader = DataLoader(dataset=data_set,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=args.prefetch,
                                 pin_memory=False,
                                 drop_last=False)
    return data_loader


def get_voc_sbd_instance_dataset(args, train, repeat):
    train_transform, val_transform = get_transform(args)

    if train:
        voc_data_set = VOCInstanceDataset(data_dir=args.voc2012_data_path, transform=train_transform, train=True,
                                          choose_size=None, repeat=repeat)
        sbd_data_set = SBDInstanceDataset(data_dir=args.sbd_data_path, transform=train_transform,
                                          choose_size=None, repeat=args.sbd_repeat)
        return ConcatDataset(datasets=[voc_data_set, sbd_data_set])
    else:
        data_set = VOCInstanceDataset(data_dir=args.voc2012_data_path, transform=val_transform, train=False)

    return data_set


def get_voc_sbd_instance_dataloader(args, train=True, repeat=1):
    assert 'balance' not in args.aug
    data_set = get_voc_sbd_instance_dataset(args, train, repeat)
    if train:
        data_loader = DataLoader(dataset=data_set,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.prefetch,
                                 pin_memory=False,
                                 drop_last=True)
    else:
        data_loader = DataLoader(dataset=data_set,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=args.prefetch,
                                 pin_memory=False,
                                 drop_last=False)
    return data_loader


if __name__ == '__main__':
    from src.utils.train_config import config
    from torchvision import transforms

    args = config()
    data_loader = get_voc_sbd_instance_dataloader(args, train=True, repeat=1)
    print(data_loader.__len__())
    for i, sample in enumerate(data_loader):
        imgs = sample['image']
        instances = sample['instance']
        labels = sample['label']
        print('label:', labels.unique())
        print('instance:', instances.unique())
        transforms.ToPILImage()(imgs[0]).show()
        transforms.ToPILImage()(instances[0] * 50).show()
        transforms.ToPILImage()(labels[0] * 50).show()
        # transforms.ToPILImage()((labels[0].squeeze() == 255) * 255).show()
        # print('255:', (labels[0].squeeze() == 255).unique())
        pass
