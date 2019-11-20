from datasets.sbd.sbd_instance_segmentation_dataset import SBDInstanceDataset
from datasets.transforms import get_transform
from torch.utils.data import DataLoader


def get_sbd_instance_dataset(args):
    train_transform, val_transform = get_transform(args)
    data_set = SBDInstanceDataset(data_dir=args.sbd_data_path, transform=train_transform,
                                  choose_size=args.sbd_data_choose_size, repeat=args.sbd_repeat)

    return data_set


def get_sbd_instance_dataloader(args):
    assert 'balance' not in args.aug
    data_set = get_sbd_instance_dataset(args)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.prefetch,
                             pin_memory=False,
                             drop_last=True)
    return data_loader


if __name__ == '__main__':
    from train_config import config
    from torchvision import transforms

    args = config()
    data_loader = get_sbd_instance_dataloader(args)
    print(data_loader.__len__())
    for i, sample in enumerate(data_loader):
        imgs = sample['image']
        instances = sample['instance']
        labels = sample['label']
        print(labels.unique())
        print(instances.unique())
        transforms.ToPILImage()(imgs[0]).show()
        transforms.ToPILImage()(instances[0] * 50).show()
        transforms.ToPILImage()(labels[0] * 50).show()

        pass

