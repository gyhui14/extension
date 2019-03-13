import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import sys
import copy
sys.path.append('../')


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return out, residual

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        return out, residual

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNetlist, self).__init__()
        self.layers = layers

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        strides = [1, 2, 2, 2]
        filt_sizes = [64, 128, 256, 512]

        # reduce dimension
        self.input_gate_layers = []

        self.inplanes = 64
        self.blocks = []    
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks = self._make_layer(block, filt_size, num_blocks, stride=stride, idx=idx)
            self.blocks.append(nn.ModuleList(blocks))

        self.blocks = nn.ModuleList(self.blocks)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, idx=0):

        layers = []

        layers.append(nn.Conv2d(self.inplanes, planes, kernel_size=1, bias=False))

        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False))

        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(planes, planes * block.expansion, kernel_size=1, bias=False))

        layers.append(nn.BatchNorm2d(planes * block.expansion))
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(planes * block.expansion))

        blocks_layers = [nn.ModuleList(layers)]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers = []
            layers.append(nn.Conv2d(self.inplanes, planes, kernel_size=1, bias=False))

            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                   padding=1, bias=False))

            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(planes, planes * block.expansion, kernel_size=1, bias=False))

            layers.append(nn.BatchNorm2d(planes * block.expansion))
            blocks_layers.append(nn.ModuleList(layers))

        return blocks_layers

    def forward(self, x, i):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for segment, num_blocks in enumerate(self.layers):
            for b in range(num_blocks):
                module_list = self.blocks[segment][b]

                residual = x
                out = x

                for idx, module in enumerate(module_list):
                    if idx >= 8:
                        residual = module(residual)
                    else:
                        out = module(out)

                x = F.relu(residual +  out)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(num_class = 10):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model = ResNet(BasicBlock, [2, 2, 2, 2], 2*num_class)
    model = ResNet(BasicBlock, [1, 1, 1, 1], 2*num_class)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetlist(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(pretrained=False, num_classes=1000):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetlist(Bottleneck, [3, 4, 6, 3], num_classes)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False, num_classes=1000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
