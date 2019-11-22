import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from util.my_util import get_inner_path
import cv2
import numpy as np
from tqdm import tqdm


def make_align_img(dir_A,dir_B,dir_AB):
    print("Align data floder creating!")
    num=0
    imgs_A=make_dataset(dir_A)
    imgs_B=make_dataset(dir_B)
    for img_A in tqdm(imgs_A):
        img_inner=get_inner_path(img_A,dir_A)
        if os.path.join(dir_B,img_inner) in imgs_B:
            photo_A=cv2.imread(img_A)
            photo_B=cv2.imread(os.path.join(dir_B,img_inner))
            if photo_A.shape==photo_B.shape:
                photo_AB=np.concatenate([photo_A, photo_B], 1)
                img_AB=os.path.join(dir_AB,img_inner)
                if not os.path.isdir(os.path.split(img_AB)[0]):
                    os.makedirs(os.path.split(img_AB)[0])
                cv2.imwrite(img_AB, photo_AB)
                num+=1
    print("Align data floder created! %d img was processed"%num)


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit![]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#todo:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
    
        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
