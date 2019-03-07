import os
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as models

import numpy as np
import shutil

from models.base import DownsampleB, conv3x3, BasicBlock
from models import resnet

import itertools
from itertools import cycle

from models.resnet_block_format import *
import models.resnet_combined_weights as cw
from models.layers import *

import models.agent_net as agent_net


from numpy import linalg as LA

def MinibatchScheduler(tloaders, mode = 'cycle'):
    if len(tloaders) == 1:
    	for i, data_pair in enumerate(tloaders[0]):
             yield i, [data_pair]
    else:
    	if mode == 'cycle':
        	ziplist = zip(tloaders[0], cycle(tloaders[1]))
        	for i in range(2, len(tloaders)):
            		flatlist = zip(*ziplist)
            		flatlist.append(cycle(tloaders[i]))
            		ziplist = zip(*flatlist)

        	for i, data_pair in enumerate(ziplist):
            		yield i, data_pair
    	elif mode == 'min':
        	for i, data_pair in enumerate(zip(*tloaders)):
            		yield i, data_pair

	elif mode == 'interleaving':
		c = [iter(i) for i in tloaders]
		i = 0
		while True:
			for task in c:
				cnt = 5
				while cnt > 0:
					yield i, [task.next()]
					cnt = cnt - 1	
					i = i + 1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0      
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def performance_stats(policies):

    policies = np.array(policies)

    #sparsity = policies.sum(1).mean()
    ave_policy = policies.mean(0)
    #variance = policies.sum(1).std()
    return ave_policy

def adjust_learning_rate_and_learning_taks(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if epoch >= args.step3:
        lr = args.lr * 0.001
    elif epoch >= args.step2:
        lr = args.lr * 0.01
    elif epoch >= args.step1:
        lr = args.lr * 0.1
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# load model weights trained using scripts from https://github.com/felixgwu/img_classification_pk_pytorch OR
# from torchvision models into our flattened resnets
def load_weights_to_flatresnet(rnet, rnet_imagenet, net_old_imagenet, net_old_place365):
    # load imagenet
    store_data_imagenet = []
    for name, m in net_old_imagenet.named_modules():
        if isinstance(m, nn.Conv2d): 
            store_data_imagenet.append(m.weight.data)

    # load place365
    store_data_places365 = []
    for name, m in net_old_place365.named_modules():
        if isinstance(m, nn.Conv2d): 
            store_data_places365.append(m.weight.data)

    element = 0
    for name, m in rnet.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.nn.Parameter( store_data_places365[element])
            element += 1

    element = 0
    for name, m in rnet_imagenet.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.nn.Parameter( store_data_imagenet[element])
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old_place365.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in rnet.named_modules():
        if isinstance(m, nn.BatchNorm2d):
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old_imagenet.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in rnet_imagenet.named_modules():
        if isinstance(m, nn.BatchNorm2d):
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    return rnet, rnet_imagenet

def load_from_pytorch_models(net_old, net, load_fc=False):
    # load pretrained net 
    store_data = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d): 
            store_data.append(m.weight.data)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel' not in name:
            m.weight.data = torch.nn.Parameter(store_data[element])
            element += 1

    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' in name:
            m.weight.data = torch.nn.Parameter(store_data[element])
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel' not in name:
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel_blocks' in name:
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    if load_fc is True:
    	net.fc.weight.data = torch.nn.Parameter(net_old.module.fc.weight.data)
    	net.fc.bias.data = torch.nn.Parameter(net_old.module.fc.bias.data)

    return net

'''
def get_model(num_classes, datasets):

    rnet = resnet50(False, num_classes)

    pretrained_model = {}

    agent = agent_net.resnet18(1, len(datasets))

    for dataset  in datasets:
        pretrained_model[dataset] = "./fine_tuned_models/" + dataset + "/" + dataset + ".t7"

    rnet = load_weights_to_flatresnet(pretrained_model, datasets, rnet)

    return rnet, agent
'''

def get_model(num_classes):
    rnet = resnet50(False, num_classes)

    net_old_imagenet = models.resnet50(pretrained=True)
    rnet = load_weights_to_flatresnet(rnet, net_old_imagenet)

    return rnet

def get_places365_model(num_classes):

    rnet = resnet50(pretrained=False, num_classes = num_classes)
    rnet_imagenet = resnet50(pretrained=False, num_classes = num_classes)

    # the architecture to use
    arch = 'resnet50'
    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    net_old_place365 = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    net_old_place365.load_state_dict(state_dict)
    
    net_old_imagenet = models.resnet50(pretrained=True)

    rnet, rnet_places365  = load_weights_to_flatresnet(rnet, rnet_imagenet, net_old_imagenet, net_old_place365)

    return rnet, rnet_imagenet
