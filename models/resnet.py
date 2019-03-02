import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import sys
sys.path.append('../')
import config_task
import math
from gumbel_softmax import *

from models.base import DownsampleB, conv3x3, BasicBlock

class FlatResNet(nn.Module):

    def seed(self, x):
        # x = self.relu(self.bn1(self.conv1(x))) -- CIFAR
        # x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) -- ImageNet
        raise NotImplementedError

    # run a variable policy batch through the resnet implemented as a full mask over the residual
    # fast to train, non-indicative of time saving (use forward_single instead)
    def forward(self, x, policy=None, task_id=0, layer_wise_agents = None):
        
        if policy is not None:
            x = self.seed(x)
            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    action = policy[:,t].contiguous()
                    residual = self.ds[segment](x) if b==0 else x
                    
                    if action.data.sum() == 0:
                        x = residual
                        t += 1
                        continue

                    action_mask = action.float().view(-1,1,1,1)
                    fx = F.relu(residual + self.blocks[segment][b](x))
                    x = fx*action_mask + residual*(1-action_mask)
                    t += 1
        else:
            x = self.seed(x)
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = self.ds[segment](x) if b==0 else x
                    x = F.relu(residual + self.blocks[segment][b](x))

        x = self.bns[task_id](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linears[task_id](x)
    
        return x

    # run a single, fixed policy for all items in the batch
    # policy is a (15,) vector. Use with batch_size=1 for profiling
    def forward_single(self, x, policy):
        x = self.seed(x)
        task = config_task.task
        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
           for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                if policy[t]==1:
                    x = residual + self.blocks[segment][b](x)
                    x = F.relu(x)
                else:
                    x = residual
                t += 1

        x = self.bns(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_full(self, x):
        x = self.seed(x)
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                x = F.relu(residual + self.blocks[segment][b](x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FlatResNet26(FlatResNet):
    def __init__(self, block, layers, num_classes = [10]):
        super(FlatResNet26, self).__init__()

        nb_tasks = len(num_classes)
        factor = config_task.factor
        self.in_planes = int(32*factor)
        self.conv1 = conv3x3(3, int(32*factor))
        self.bn1 = nn.BatchNorm2d(int(32*factor))

        strides = [2, 2, 2]
        filt_sizes = [64, 128, 256]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)
        
        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

	self.bns = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(int(256*factor)), nn.ReLU(True)) for i in range(nb_tasks)])
        self.avgpool = nn.AdaptiveAvgPool2d(1) 
        
        self.linears = nn.ModuleList([nn.Linear(int(256*factor), num_classes[i]) for i in range(nb_tasks)])         

        self.layer_config = layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.conv1(x)
	x = self.bn1(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = DownsampleB(self.in_planes, planes * block.expansion, 2)

        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return layers, downsample

class FlatResNet50(FlatResNet):
    def __init__(self, block, layers, num_classes = [10]):
        super(FlatResNet50, self).__init__()

        nb_tasks = len(num_classes)
        factor = config_task.factor
        self.in_planes = int(32*factor)
        self.conv1 = conv3x3(3, int(32*factor))
        self.bn1 = nn.BatchNorm2d(int(32*factor))

        strides = [2, 2, 2, 2]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)
        
        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.bn2 = nn.BatchNorm2d(int(filt_sizes[-1]*factor))
        self.avgpool = nn.AdaptiveAvgPool2d(1) 
        
        self.linears = nn.ModuleList([nn.Linear(int(filt_sizes[-1]*factor), num_classes[i]) for i in range(nb_tasks)])         

        self.layer_config = layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.conv1(x)
    	x = self.bn1(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = DownsampleB(self.in_planes, planes * block.expansion, 2)

        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return layers, downsample

#---------------------------------------------------------------------------------------------------------#
# Class to generate resnetNB or any other config (default is 3B)
# removed the fc layer so it serves as a feature extractor
class Policy26(nn.Module):

    def __init__(self, layer_config=[1,1,1], blocks = 12, num_class = 10):
        super(Policy26, self).__init__()
        self.features = FlatResNet26(BasicBlock, layer_config, num_classes=[10])
        self.feat_dim = self.features.linears[0].weight.data.shape[1]
        self.features.fc = nn.Sequential()
	
        self.logit2 = nn.Linear(self.feat_dim, blocks)

    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy26, self).load_state_dict(state_dict)

    def forward(self, x, logits = None):
        x = self.features.forward_full(x)
        if logits is not None:
        	x = torch.cat((x, logits), 1)

        probs = self.logit2(x) 
	return probs

'''
class Policy224(nn.Module):

    def __init__(self, layer_config=[1,1,1,1], num_blocks=16):
        super(Policy224, self).__init__()
        self.features = FlatResNet224(base.BasicBlock, layer_config, num_classes=1000)

        resnet18 = torchmodels.resnet18(pretrained=True)
        utils.load_weights_to_flatresnet(resnet18, self.features)

        self.features.avgpool = nn.AvgPool2d(4)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = nn.Sequential()


        self.logit = nn.Linear(self.feat_dim, num_blocks)
        self.vnet = nn.Linear(self.feat_dim, 1)

    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy224, self).load_state_dict(state_dict)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = self.features.forward_full(x)
        value = self.vnet(x)
        probs = F.sigmoid(self.logit(x))
        return probs, value
#--------------------------------------------------------------------------------------------------------#

class StepResnet32(FlatResNet32):

    def __init__(self, block, layers, num_classes, joint=False):
        super(StepResnet, self).__init__(block, layers, num_classes)
        self.eval() # default to eval mode

        self.joint = joint

        self.state_ptr = {}
        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                self.state_ptr[t] = (segment, b)
                t += 1

    def seed(self, x):
        self.state = self.relu(self.bn1(self.conv1(x)))
        self.t = 0
'''
