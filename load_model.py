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

import sys
sys.path.append('../')
import sgd
import config_task
import imdbfolder_coco as imdbfolder
from piggy_back_loader import *

from gumbel_softmax import *
from imagenet_loader import *

from utils import *
from models.resnet_block_format import *

idx=  range(30, 44)

for ii in idx:
    source = "./CUBS/stanford_cars/stanford_cars_epoch_" + str(ii) + ".t7"
    checkpoint = torch.load(source)
    print checkpoint["acc"]
