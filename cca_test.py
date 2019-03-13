import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import os
import time
import argparse
import numpy as np
import json
from torch.autograd import Variable
import collections
import copy

from utils import *

cca_loss = cca_loss(outdim_size=1, use_all_singular_values=True, device=torch.device('cuda'))


A_fake = np.random.randn(100, 200)

A_fake = Variable(torch.from_numpy(A_fake).float().cuda())

weight = Variable(torch.from_numpy(np.random.randn(100, 100)).float().cuda())

At_fake = Variable(torch.matmul(weight, A_fake)).float().cuda()

print (cca_loss.loss(A_fake, At_fake))