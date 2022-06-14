import os
import torch
from torch import nn
import torch.nn.functional as F
from base import BaseModel
import math
import numpy as np
from .resnet import resnet50_locate 

class TopDownLayer(nn.Module):
    def __init__(self, center, need_x2, need_fuse):
        super(TopDownLayer, self).__init__()

        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        self.conv = nn.Sequential(nn.Conv2d(center, center, 3, 1, 1, bias=False), nn.BatchNorm2d(center))
        self.relu = nn.ReLU()
        self.conv_sum = nn.Sequential(nn.Conv2d(center, center, 3, 1, 1, bias=False), nn.BatchNorm2d(center))
        if self.need_fuse:
            self.conv_sum_c = nn.Sequential(nn.Conv2d(center, center, 1, 1, bias=False), nn.BatchNorm2d(center))

    def forward(self, x, x2=None, x3=None):
        x_size = x.size()
        resl = self.conv(x)
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(resl, x2))
        return resl

class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k ,1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

class CII(BaseModel):
    def __init__(self, base, convert, center, topdown, score, pretrained=None):
        super(CII, self).__init__()
        self.base = resnet50_locate(convert, center)

        self.conv = nn.Sequential(nn.Conv2d(center, center, 3, 1, 1, bias=False), nn.BatchNorm2d(center), nn.ReLU(inplace=True))
        self.conv_sum = nn.Sequential(nn.Conv2d(center, center, 3, 1, 1, bias=False), nn.BatchNorm2d(center))
        self.conv_sum_c = nn.Sequential(nn.Conv2d(center, center, 1, 1, bias=False), nn.BatchNorm2d(center))

        self.topdown = nn.ModuleList()
        for i in range(len(topdown[0])):
            self.topdown.append(TopDownLayer(center, topdown[0][i], topdown[1][i]))
        self.score = ScoreLayer(score)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if pretrained is not None:
            assert os.path.exists(pretrained), '{} does not exist.'.format(pretrained)
            print("Loading pretrained parameters from {}.".format(pretrained))
            self.base.load_pretrained_model(torch.load(pretrained))

    def forward(self, x):
        x_size = x.size()
        infos = self.base(x)
        infos = infos[::-1]

        merge = self.conv_sum_c(self.conv_sum(self.conv(infos[0])))
        for k in range(len(infos)-1):
            merge = self.topdown[k](merge, infos[k+1])

        merge = self.topdown[-1](merge)
        merge = self.score(merge, x_size)
        return merge