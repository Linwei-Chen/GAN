import numpy as np
import os
import warnings

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.sbd import sbd_utils
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image
from torch.utils.data import Dataset
from PIL import Image
from datasets.voc.voc_instance_segmentation_dataset import choose_uniformly

try:
    import scipy

    _available = True
except ImportError:
    _available = False


def _check_available():
    if not _available:
        warnings.warn(
            'SciPy is not installed in your environment,'
            'so the dataset cannot be loaded.'
            'Please install SciPy to load dataset.\n\n'
            '$ pip install scipy')


class SBDInstanceSegmentationDataset(GetterDataset):
    """Instance segmentation dataset for Semantic Boundaries Dataset `SBD`_.

    .. _`SBD`: http://home.bharathh.info/pubs/codes/SBD/download.html

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/sbd`.
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

    def __init__(self, data_dir='auto', split='train_ex_voc2012', choose_size=None, repeat=1):
        super(SBDInstanceSegmentationDataset, self).__init__()

        _check_available()

        # if split not in ['train', 'trainval', 'val']:
        #     raise ValueError(
        #         'please pick split from \'train\', \'trainval\', \'val\'')
        #
        if data_dir == 'auto':
            data_dir = sbd_utils.get_sbd()

        # id_list_file = os.path.join(
        #     data_dir, '{}_voc2012.txt'.format(split))
        id_list_file = os.path.join(
            data_dir, '{}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        if choose_size is not None:
            self.ids = choose_uniformly(image_list=self.ids, choose_size=choose_size)
        self.ids = self.ids * repeat

        self.data_dir = data_dir

        self.add_getter('img', self._get_image)
        self.add_getter(('mask', 'label'), self._get_annotations)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        data_id = self.ids[i]
        img_file = os.path.join(
            self.data_dir, 'img', data_id + '.jpg')
        return read_image(img_file, color=True)

    def _get_image_file(self, i):
        data_id = self.ids[i]
        img_file = os.path.join(
            self.data_dir, 'img', data_id + '.jpg')
        return img_file

    def _get_annotations(self, i):
        data_id = self.ids[i]
        label_img, inst_img = self._load_label_inst(data_id)
        mask, label = voc_utils.image_wise_to_instance_wise(
            label_img, inst_img)
        return mask, label

    def _load_label_inst(self, data_id):
        label_file = os.path.join(
            self.data_dir, 'cls', data_id + '.mat')
        inst_file = os.path.join(
            self.data_dir, 'inst', data_id + '.mat')
        label_anno = scipy.io.loadmat(label_file)
        label_img = label_anno['GTcls']['Segmentation'][0][0].astype(np.int32)
        inst_anno = scipy.io.loadmat(inst_file)
        inst_img = inst_anno['GTinst']['Segmentation'][0][0].astype(np.int32)
        inst_img[inst_img == 0] = -1
        inst_img[inst_img == 255] = -1
        return label_img, inst_img


class SBDInstanceDataset(Dataset):
    def __init__(self, data_dir, transform=None, choose_size=None, repeat=1):
        self.data_set = SBDInstanceSegmentationDataset(data_dir, choose_size=choose_size, repeat=repeat)
        self.transform = transform

    def __len__(self):
        return self.data_set.__len__()

    def __getitem__(self, index):
        try:
            img = self.data_set._get_image(index).astype(np.uint8).transpose(1, 2, 0)
            mask, label_vec = self.data_set._get_annotations(index)
            label_vec += 1
            # print(label_vec)
            label = np.zeros(mask.shape[1:]).astype(np.uint8)
            inst = np.zeros(mask.shape[1:]).astype(np.uint8)
            inst_count = 1
            sample = {}
            for m, l in zip(mask, label_vec):
                inst[m] = inst_count
                label[m] = l
                inst_count += 1
            # print(np.unique(inst))
            sample['instance'] = Image.fromarray(inst)
            sample['label'] = Image.fromarray(label)
            sample['smask'] = Image.fromarray(label)
            sample['image'] = Image.fromarray(img)
            sample['im_name'] = self.data_set._get_image_file(index)
            # sample['im_name'] = ''
            if self.transform is not None:
                return self.transform(sample)
            else:
                return sample
        except Exception:
            return self.__getitem__(np.random.randint(self.__len__()))


if __name__ == '__main__':
    from train_config import config

    args = config()
    dataset = SBDInstanceDataset(data_dir=args.sbd_data_path)
    sample = dataset.__getitem__(101)
    print(sample['im_name'])
    sample['image'].show()
    Image.fromarray(np.array(sample['instance']) * 40).show()
    Image.fromarray(np.array(sample['label']) * 40).show()
