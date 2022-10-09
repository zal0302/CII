import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, is_last=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, is_last=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out_mid = out

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.is_last:
            return out, out_mid
        return out

    
class ResNet(nn.Module):

    def __init__(self, block, layers,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        self.freeze_bn = True
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # is_last = True if (_ == blocks-1) else False
            is_last = False
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=dilation,
                                norm_layer=norm_layer, is_last=is_last))

        return nn.Sequential(*layers)

    def load_pretrained_model(self, model):
        self.load_state_dict(model, strict=False)

    # def train(self, mode=True):
    #     """
    #     Override the default train() to freeze the BN parameters
    #     """
    #     super(ResNet, self).train(mode)
    #     if self.freeze_bn:
    #         print("Freezing Mean/Var of BatchNorm2D.")
    #         print("Freezing Weight/Bias of BatchNorm2D.")
    #         for m in self.modules():
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m.eval()
    #                 m.weight.requires_grad = False
    #                 m.bias.requires_grad = False

    def _forward_impl(self, x):
        tmp_x = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_x.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        tmp_x.append(x)
        x = self.layer2(x)
        tmp_x.append(x)
        x = self.layer3(x)
        tmp_x.append(x)
        x = self.layer4(x)
        tmp_x.append(x)


        return tmp_x

    def forward(self, x):
        return self._forward_impl(x)

    
class Basic2(nn.Module):
    def __init__(self):
        super(Basic2,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
 
    def forward(self, x):
        x = self.conv1(x) + x
        return x
    def initialize(self):
        weight_init(self)

        
class RGC(nn.Module):
    def __init__(self, channel, ratio=4):
        super(RGC, self).__init__()
        self.basic1 = Basic2()
        self.basic2 = Basic2()
        self.basic3 = Basic2()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.basic1(x1)
        x2 = self.basic2(x2)
        x2 = self.sigmoid(self.max_pool(x2))       
        
        return self.basic3(x2 * x1) 

    
class ResNet_locate(nn.Module):
    def __init__(self, block, layers, convert, center):
        super(ResNet_locate,self).__init__()
        self.resnet = ResNet(block, layers)
        self.down = nn.Sequential(nn.Conv2d(center, center, 3, 1, 1, bias=False), nn.BatchNorm2d(center), nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.in_planes = convert
        
        rgc_pre = []
        for inplane in self.in_planes:
            rgc_pre.append(nn.Sequential(nn.Conv2d(inplane, center, 1, 1, bias=False), nn.BatchNorm2d(center), nn.ReLU(inplace=True)))
        self.rgc_pre = nn.ModuleList(rgc_pre)
        self.RGC = RGC(center)

    def load_pretrained_model(self, model):
        self.resnet.load_state_dict(model, strict=False)

    def forward(self, x):
        x_size = x.size()[2:]
        xs = self.resnet(x)
        for i in range(5):
            xs[i] = self.rgc_pre[i](xs[i])

        xls = []    
        for i in range(len(self.in_planes)-1):
            xls.append(self.RGC(xs[i], xs[i+1]))
        xls.append(self.RGC(xs[i+1], self.down(xs[i+1])))
        
        return xls


def resnet50_locate(convert, center):
    model = ResNet_locate(Bottleneck, [3, 4, 6, 3], convert, center)
    return model


def resnet18_locate(convert, center):
    model = ResNet_locate(BasicBlock, [2, 2, 2, 2], convert, center)
    return model
