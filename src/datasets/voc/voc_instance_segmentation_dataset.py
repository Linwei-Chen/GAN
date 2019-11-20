import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from datasets.voc import voc_utils
from chainercv.utils import read_image
from chainercv.utils import read_label
from torch.utils.data import Dataset
from PIL import Image


def choose_uniformly(image_list, choose_size):
    size = len(image_list)
    mask = list(range(size))
    if choose_size is not None:
        choose_rest = False
        if choose_size < 0:
            choose_rest = True
            choose_size = abs(choose_size)

        interval = size // choose_size
        mask = list(range(size))[0::interval][:choose_size]
        if choose_rest:
            temp = []
            for i in range(size):
                if i not in mask:
                    temp.append(i)
            mask = temp
    return [image_list[i] for i in mask]


class VOCInstanceSegmentationDataset(GetterDataset):
    """Instance segmentation dataset for PASCAL `VOC2012`_.

    .. _`VOC2012`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`mask`, ":math:`(R, H, W)`", :obj:`bool`, --
        :obj:`label`, ":math:`(R,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
    """

    def __init__(self, data_dir='auto', split='train', choose_size=None, repeat=1):
        super(VOCInstanceSegmentationDataset, self).__init__()

        if split not in ['train', 'trainval', 'val']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = voc_utils.get_voc('2012', split)

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        if choose_size is not None:
            self.ids = choose_uniformly(self.ids, choose_size)
        self.ids = self.ids * repeat

        self.data_dir = data_dir

        self.add_getter('img', self._get_image)
        self.add_getter(('mask', 'label'), self._get_annotations)
        print('===> VOC2012:')
        print(f'data_dir={data_dir} | split={split} | choose_size={choose_size} | repeat={repeat}')

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        data_id = self.ids[i]
        img_file = os.path.join(
            self.data_dir, 'JPEGImages', data_id + '.jpg')
        # return np.array(read_image(img_file, color=True)).astype(np.uint8)
        return read_image(img_file, color=True)

    def _get_image_file(self, i):
        data_id = self.ids[i]
        img_file = os.path.join(
            self.data_dir, 'JPEGImages', data_id + '.jpg')
        return img_file

    def _get_annotations(self, i):
        """
        :param i:
        :return: np.array[obj_num, h, w], np.array(obj_num)
        """
        data_id = self.ids[i]
        label_img, inst_img = self._load_label_inst(data_id)
        mask, label = voc_utils.image_wise_to_instance_wise(
            label_img, inst_img)
        # return np.array(mask), np.array(label)
        return mask, label

    def _load_label_inst(self, data_id):
        label_file = os.path.join(
            self.data_dir, 'SegmentationClass', data_id + '.png')
        inst_file = os.path.join(
            self.data_dir, 'SegmentationObject', data_id + '.png')

        label_img = read_label(label_file, dtype=np.int32)
        # label_img[label_img == 255] = -1
        inst_img = read_label(inst_file, dtype=np.int32)
        inst_img[inst_img == 0] = -1
        # inst_img[inst_img == 255] = -1
        return label_img, inst_img


class VOCInstanceDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True, choose_size=None, repeat=1):
        split = ('train' if train else 'val')
        self.data_set = VOCInstanceSegmentationDataset(data_dir, split, choose_size, repeat)
        self.transform = transform

    def __len__(self):
        return self.data_set.__len__()

    def __getitem__(self, index):
        img = self.data_set._get_image(index).astype(np.uint8).transpose(1, 2, 0)
        mask, label_vec = self.data_set._get_annotations(index)
        # 20191027 modified the datasets.voc.voc_utils, labels starts with 1, keep the 255
        label_vec += 1
        label = np.zeros(mask.shape[1:]).astype(np.uint8)
        label_255 = np.zeros(mask.shape[1:]).astype(np.uint8)
        inst = np.zeros(mask.shape[1:]).astype(np.uint8)
        inst_count = 1
        sample = {}
        for m, l in zip(mask, label_vec):
            if l == np.array(255):
                # only l
                # label_255[m] = 1
                label[m] = l
                inst[m] = 0
            else:
                inst[m] = inst_count
                label[m] = l
                inst_count += 1

        sample['instance'] = Image.fromarray(inst)
        sample['label'] = Image.fromarray(label)
        sample['smask'] = Image.fromarray(label)
        # sample['label_255'] = Image.fromarray(label_255)
        sample['im_name'] = self.data_set._get_image_file(index)
        sample['image'] = Image.fromarray(img)
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample


def _test_VOCInstanceSegmentationDataset():
    dataset = VOCInstanceSegmentationDataset(data_dir='/Users/chenlinwei/dataset/VOCdevkit/VOC2012')
    label, inst = dataset._get_annotations(100)
    print(label)
    print(inst)
    Image.fromarray(np.array(label[0]).astype(np.uint8) * 100).show()
    img = dataset._get_image(100).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(img).show()
    pass


if __name__ == '__main__':
    dataset2 = VOCInstanceDataset(data_dir='/Users/chenlinwei/dataset/VOCdevkit/VOC2012',
                                  train=True, choose_size=800, repeat=2)
    print(dataset2.__len__())
    sample = dataset2.__getitem__(101)
    print(sample['im_name'])
    # sample = dataset2.__getitem__(100)
    # for k in sample:
    #     sample[k].show()
