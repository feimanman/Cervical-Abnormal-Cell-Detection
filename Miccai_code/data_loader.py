from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from PIL import Image

import cfg
import cv2

class CervicalDataset(Dataset):
    """Cervical dataset."""

    def __init__(self, data_path, patch_size, transform=None):
        self.data_path = data_path
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img, label,path = self.load_data(idx)
        sample = {'img': img, 'label': label}


        if self.transform:
            sample = self.transform(sample)
        # x=sample['img']
        # x= x * 255
        # cv2.imwrite('0.jpg', x)
        sample['path']=path

        return sample

    def load_data(self, index):
        data = np.load(self.data_path[index])

        img, label = data['img'], data['label']/1.0
        img = img.astype(np.float32) / 255.0
        path=self.data_path[index].split('/')[-1][:-4]



        return img, label,path


class WsiDataset(Dataset):
    def __init__(self, read, y_num, x_num, strides, coordinates, patch_size, transform=None):
        self.read = read
        self.y_num = y_num
        self.x_num = x_num
        self.strieds = strides
        self.coordinates = coordinates
        self.patch_size = patch_size
        self.transform = transform

    def __getitem__(self, index):
        coord_y, coord_x = self.coordinates[index]
        img = self.read.ReadRoi(coord_x, coord_y, cfg.patch_size[0], cfg.patch_size[1], scale=20).copy()

        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32) / 255.0

        return torch.from_numpy(img).float(), coord_y, coord_x

    def __len__(self):
        return self.y_num * self.x_num


def collater(data):
    imgs = [s['img'] for s in data]
    labels = [s['label'] for s in data]
    path=[s['path'] for s in data]
    imgs = torch.stack(imgs, dim=0)
    imgs = imgs.permute((0, 3, 1, 2))

    max_num_labels = max(label.shape[0] for label in labels)

    if max_num_labels > 0:
        label_pad = torch.ones((len(labels), max_num_labels, 4)) * -1

        for idx, label in enumerate(labels):
            if label.shape[0] > 0:
                label_pad[idx, :label.shape[0], :] = label
    else:
        label_pad = torch.ones((len(labels), 1, 4)) * -1

    return {'img': imgs, 'label': label_pad,'path':path}


class CervicalDataset1(Dataset):
    """Cervical dataset."""

    def __init__(self, data_path, patch_size, transform=None):
        self.data_path = data_path
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img, label = self.load_data(idx)
        # sample = {'img': img, 'label': label}
        # print('img_shape',img.shape)

        # Do data augmentation
        # img = img * 255
        # # img = cv2.cvtColor(
        # #     img, cv2.COLOR_BGR2RGB
        # # )
        # if img.shape != (1024,1024,3):
        #     img = img.transpose(1, 0, 2)
        if self.transform:
            # sample = self.transform(sample)
            img=self.transform(image=img)['image']
        # img=torch.from_numpy(img)
        # print('img', img)
        # img= img * 255
        # cv2.imwrite('0.jpg', img)
        sample = {'img': torch.from_numpy(img).float(), 'label': torch.from_numpy(label)}

        return sample

    def load_data(self, index):
        data = np.load(self.data_path[index])
        img, label = data['img'], data['label'] / 1.0
        img = img.astype(np.float32) / 255.0

        return img, label

if __name__ == '__main__':
    sample_path = [os.path.join('/mnt/data/feimanman/npzdata/ce/', x) for x in os.listdir('/mnt/data/feimanman/npzdata/ce/') if '.npz' in x]
    train_data = CervicalDataset1(sample_path, cfg.patch_size)
    svpath='/mnt/data/feimanman/global_local_net/code3/data/'

    for i in range(train_data.__len__()):
        sample=train_data.__getitem__(i)

        x=sample['img']
        x= x * 255
        x=np.array(x)
        cv2.imwrite(svpath+'%03d.jpg'%(i), x)