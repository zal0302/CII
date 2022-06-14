import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.transforms import functional as tvF
from base import BaseDataLoader

class SodDataset(data.Dataset):
    """
    SOD dataset
    """
    def __init__(self, data_dir, data_list, trsfms=None, trsfm=None, target_trsfm=None):
        self.data_dir = data_dir
        with open(data_list, 'r') as f:
            self.data_list = [x.strip() for x in f.readlines()]
        self.data_num = len(self.data_list)
        self.trsfms = trsfms
        self.trsfm = trsfm
        self.target_trsfm = target_trsfm

    def __getitem__(self, item):
        if len(self.data_list[item].split()) == 2:
            image_name = self.data_list[item].split()[0]
            target_name = self.data_list[item].split()[1]
        else:
            image_name = self.data_list[item].split()[0]
            target_name = image_name[:-4] + '.png'

        image = Image.open(os.path.join(self.data_dir, image_name)).convert('RGB') 
        image_size = torch.Tensor(image.size[::-1])

        target = Image.open(os.path.join(self.data_dir, target_name)).convert('L') 

        if self.trsfms is not None:
            image, target = self.trsfms(image, target)
        if self.trsfm is not None:
            image = self.trsfm(image)
        if self.target_trsfm is not None:
            target = self.target_trsfm(target)

        return image, target, image_name, image_size

    def _multiscale_collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, target, image_name, image_size = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = F.interpolate(image[i].unsqueeze(0), (size, size), mode='bilinear', align_corners=True)[0]
            target[i] = F.interpolate(target[i].unsqueeze(0), (size, size), mode='bilinear', align_corners=True)[0]
            # image[i] = tvf.to_tensor(tvF.resize(image[i], (size, size), 'PIL.Image.BILINEAR'))
            # target[i] = tvf.to_tensor(tvF.resize(target[i], (size, size), 'PIL.Image.BILINEAR'))
        image = torch.stack(image, 0)
        target = torch.stack(target, 0)
        return image, target, image_name, image_size

    def __len__(self):
        return self.data_num
