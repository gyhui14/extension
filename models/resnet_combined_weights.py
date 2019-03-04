import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import sys
import copy

import layers as nl

sys.path.append('../')
from gumbel_softmax import *

def conv3x3(in_planes, out_planes, mask_init, mask_scale, threshold_fn, threshold ,stride=1):
    "3x3 convolution with padding"
    return nl.ElementWiseConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                padding=1, bias=False, mask_init=mask_init, mask_scale=mask_scale,
                                threshold_fn=threshold_fn, threshold=threshold)

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

    def __init__(self, inplanes, planes, mask_init, mask_scale, threshold_fn, threshold ,stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nl.ElementWiseConv2d(
            inplanes, planes, kernel_size=1, bias=False,
            mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nl.ElementWiseConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nl.ElementWiseConv2d(
            planes, planes * 4, kernel_size=1, bias=False,
            mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold)

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

    def __init__(self, block, layers, mask_init, mask_scale, threshold_fn, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
     
        self.threshold = nn.Parameter(torch.Tensor([0.0]), requires_grad = False)

        self.layers = layers

        self.conv1 = nl.ElementWiseConv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False,
            mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=self.threshold)
    
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        strides = [1, 2, 2, 2]
        filt_sizes = [64, 128, 256, 512]

        self.blocks = []    
        self.inplanes = 64
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks = self._make_layer(block, filt_size, num_blocks, mask_init, mask_scale, threshold_fn, self.threshold, stride=stride)
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

    def _make_layer(self, block, planes, blocks, mask_init, mask_scale, threshold_fn, threshold, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nl.ElementWiseConv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False,
                    mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, mask_init, mask_scale, threshold_fn, threshold, stride, downsample)]

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, mask_init,mask_scale, threshold_fn, threshold))
        return layers

    def forward(self, x):
        t = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for segment, num_blocks in enumerate(self.layers):
                for b in range(num_blocks):
                    output, residual = self.blocks[segment][b](x)
                    x = F.relu(residual + output)
    
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
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(mask_init='uniform', mask_scale=1e-2, threshold_fn='binarizer', pretrained=False, num_classes=1000):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], mask_init, mask_scale, threshold_fn, num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
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
