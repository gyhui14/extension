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
from gumbel_softmax import *


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size = 2048):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes

        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                )
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.ReLU(),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layers = layers

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        strides = [1, 2, 2, 2]
        filt_sizes = [64, 128, 256, 512]

        self.blocks = []    
        self.inplanes = 64
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))

        self.blocks = nn.ModuleList(self.blocks)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        ################################################################
        self.adapt_avgpool = nn.MaxPool2d(5, stride=2)

        self.dim_reduction = []
        self.dim_reduction.append(nn.Conv2d(64, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(256, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(256, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(256, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(512, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(512, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(512, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(512, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(1024, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(1024, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(1024, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(1024, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(1024, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(1024, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(2048, 32, kernel_size=1, bias=False))
        self.dim_reduction.append(nn.Conv2d(2048, 32, kernel_size=1, bias=False))
        self.dim_reduction = nn.ModuleList(self.dim_reduction)


        #self.fc_adapt_students = nn.Linear(512 * block.expansion, 32)
        #self.fc_adapt_teachers = nn.Linear(512 * block.expansion, 32)

        #self.fc_adapt_student = nn.Linear(512 * block.expansion, 32)
        #self.fc_adapt_teacher = nn.Linear(512 * block.expansion, 32)
        #self.fc_adapt_student = MlpNet([512, 64], 512 * block.expansion)
        #self.fc_adapt_teacher = MlpNet([512, 64], 512 * block.expansion)
        ################################################################

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers

    def forward(self, x, output_teacher=None, skip = False):
        t = 0
        middle_outputs = []


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        '''
        if skip is True:
            print self.dim_r[0].weight.data
        '''
        for segment, num_blocks in enumerate(self.layers):
                for b in range(num_blocks):
                    x_tmp = self.adapt_avgpool(x)
                    x_tmp = self.dim_reduction[t](x_tmp)
                    t += 1
                    num_datapoints, c, h, w = x_tmp.size()
                    x_tmp = x_tmp.reshape(num_datapoints*w*h, c)
                    middle_outputs.append(x_tmp)

                    output, residual = self.blocks[segment][b](x)
                    x = F.relu(residual + output)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if skip is True:
            '''
            x_norm = torch.norm(x, p=2).detach()
            x_normalized = x.div(x_norm)
            '''
            #x = self.fc_adapt_teacher(x)
            return middle_outputs
            
        else:
            x_output = self.fc(x)
            #x_adapt = self.fc_adapt_student(x)
            
            '''
            x_adapt_norm = torch.norm(x_adapt, p=2).detach()
            x_adapt_normalized = x_adapt.div(x_adapt_norm)
            '''
            return x_output, middle_outputs
        

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

def resnet50(pretrained=False, num_classes=1000):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
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
