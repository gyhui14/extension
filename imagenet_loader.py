import argparse
import numpy
import os
import shutil
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
  
# Data loading code
#traindir = os.path.join("/diva09/ImageNet/", 'train')
#valdir = os.path.join("/diva09/ImageNet/", 'val')

traindir = os.path.join("/home/yunhui/ImageNet/", 'train')
valdir = os.path.join("/home/yunhui/ImageNet/", 'val')
# Normalize on RGB Value
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

size = (224, 256)
cuda = torch.cuda.is_available()
# Pin memory
if cuda:
    pin_memory = True
else:
    pin_memory = False

def get_ImageNet_Loader():
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
        traindir,
        transforms.Compose([
        transforms.RandomSizedCrop(size[0]), #224 , 299
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])), 
        batch_size=128, shuffle=True,
        num_workers=32, pin_memory=pin_memory)
    
    # Validate -> Preprocessing -> Tensor
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(size[1]), # 256
            transforms.CenterCrop(size[0]), # 224 , 299
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=32, pin_memory=pin_memory)
    return train_loader, val_loader
