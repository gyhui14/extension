import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.utils as torchutils
from torch.nn import init, Parameter

import sys
sys.path.append('../')
import config_task

class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride=2):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)

    def forward(self, x):
        residual = self.avg(x)
        return torch.cat((residual, residual*0),1)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# No projection: identity shortcut
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, nb_tasks=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

	self.conv2 = nn.Sequential(nn.ReLU(True), conv3x3(planes, planes))
        self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])

    def forward(self, x):
        task = config_task.task
        out = self.conv1(x)
        out = self.bn1(out)
	out = F.relu(out)
        out = self.conv2(out)
        y = self.bns[task](out)
        return y

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedforwardNN, self).__init__()
        # reduction rate
        d = 1
        hidden_size = input_size / d
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def get_layer_wise_agents():
    filt_sizes = [64, 128, 256]
    layers = [4, 4, 4]

    action_networks = []
    action_network = FeedforwardNN(32, 2)
        
    action_networks.append(action_network)
    for idx, (filt_size, num_blocks) in enumerate(zip(filt_sizes, layers)):
        for _ in range(num_blocks):
            action_network = FeedforwardNN(filt_size, 2)
            action_networks.append(action_network)
    action_networks = action_networks[:-1]
    return action_networks

def get_agents(loaders):
    agents = dict()
    for task_id in range(len(loaders)):
        agents[task_id] = FeedforwardNN(len(loaders), 24)

    return agents

