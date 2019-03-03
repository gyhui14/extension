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

import sys
sys.path.append('../')

import sgd
import config_task
import imdbfolder_coco as imdbfolder
from piggy_back_loader import * 
from sun_loader import *

from gumbel_softmax import *
from imagenet_loader import *

from utils import *
from models.resnet_block_format import *
from dog_data.load import load_datasets

parser = argparse.ArgumentParser(description='PyTorch Residual Adapters training')
parser.add_argument('--nb_epochs', default=60, type=int, help='nb epochs')

parser.add_argument('--wd3x3', default=1.0, type=float, nargs='+', help='weight decay for the 3x3')
parser.add_argument('--wd1x1', default=1.0, type=float, nargs='+', help='weight decay for the 1x1')

parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--lr_agent', default=1e-1, type=float, help='initial learning rate')

parser.add_argument('--batch_size', default=32, type=int, help="batch size")

parser.add_argument('--lambd', default=2, type=float, help='balance loss and fire rate')
parser.add_argument('--factor', default='1.', type=float, help='Width factor of the network')

parser.add_argument('--datadir', default='../data/decathlon-1.0/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='../data/decathlon-1.0/annotations/', help='annotation folder')
parser.add_argument('--cv_dir', default='./fine_tuned_models/', help='checkpoint directory (models and logs are saved here)')
#parser.add_argument('--cv_dir', default='./test/', help='checkpoint directory (models and logs are saved here)')

parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--step1', default=15, type=int, help='nb epochs before first lr decrease')
parser.add_argument('--step2', default=30, type=int, help='nb epochs before second lr decrease')
parser.add_argument('--step3', default=45, type=int, help='nb epochs before third lr decrease')
args = parser.parse_args()

weight_decays = [
    ("cubs", 0.0),
    ("stanford_cars", 0.0),
    ("flowers", 0.0),
    ("wikiart", 0.0),
    ("sketches", 0.0)]

dataset_classes = [
    ("cubs", 200),
    ("stanford_cars", 196),
    ("flowers", 102),
    ("wikiart", 195),
    ("sketches", 250)]

datasets = [
   #("cubs", 0),
   #("wikiart", 1),
   ("flowers", 0)]
   #("sketches", 3),
   #("stanford_cars", 4)]

Resnet_type = "Resnet50"

config_task.decay3x3 = np.array(args.wd3x3) * 0.0
config_task.decay1x1 = np.array(args.wd1x1) * 0.0

datasets = collections.OrderedDict(datasets)
dataset_classes = collections.OrderedDict(dataset_classes)
weight_decays = collections.OrderedDict(weight_decays)


def train(epoch, train_loaders, net, net_optimizer):
    #Train the model
    net.train()

    total_step = len(train_loaders)
    top1 = AverageMeter()
    losses = AverageMeter()

    for i, (images, labels) in enumerate(train_loaders):
        if use_cuda:
            images, labels = images.cuda(async=True), labels.cuda(async=True)
        
        images, labels = Variable(images), Variable(labels)
        outputs = net.forward(images)

        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))	    

        # Loss
        loss = criterion(outputs, labels)
        losses.update(loss.data[0], labels.size(0))

        if i % 10 == 0:
            print('Epoch [{}/{}]\t'
                  'Batch [{}/{}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(
                   epoch, args.nb_epochs, i, total_step,
                   loss=losses,top1=top1))
    
        #---------------------------------------------------------------------#
        # Backward and optimize            
        net_optimizer.zero_grad()
        loss.backward()  
        net_optimizer.step()

        del loss
        
    return top1.avg, losses.avg

def test(epoch, val_loaders, net, best_acc, dataset):

    net.eval()

    top1 = AverageMeter()
    losses = AverageMeter()

    # Test the model
    with torch.no_grad():

        for i, (images, labels) in enumerate(val_loaders):

            if use_cuda:
                images, labels = images.cuda(async=True), labels.cuda(async=True)          

            images, labels = Variable(images), Variable(labels)

            outputs = net.forward(images)

            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(labels.data).cpu().sum()
            top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))

            #Loss
            loss = criterion(outputs, labels)
            losses.update(loss.data[0], labels.size(0))
        
        print "test accuracy"
        print('Epoch [{}/{}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(
           epoch, args.nb_epochs,
           loss=losses, top1=top1))	

    acc = top1.avg
    # Save checkpoint
    if acc > best_acc:
    	print('Saving..')
    	state = {
            'net': net,
    	    'acc': acc,
    	    'epoch': epoch,
	    }   

        torch.save(state, pretrained_model_dir + '/' + dataset + '.t7')
        best_acc = acc
    
    return top1.avg, losses.avg, best_acc

#####################################
# Prepare data loaders
criterion = nn.CrossEntropyLoss().cuda()
np.random.seed(args.seed)

for i, dataset in enumerate(datasets.keys()):

    '''
    print dataset 
    if dataset in ['flowers', 'wikiart', 'skethes']:
    	train_loaders = train_loader("../cubs_data/" + dataset +"/train/", 64)
    	val_loaders = test_loader("../cubs_data/" + dataset + "/test/", 64)
    else:
    	train_loaders = train_loader_cropped("../cubs_data/" + dataset +"/train/", 64)
    	val_loaders = test_loader_cropped("../cubs_data/" + dataset + "/test/", 64)
    '''

    '''
    dataset = "cifar100"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    train_loaders = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)

    val_loaders = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=4)
    '''

    '''
    dataset = "caltech_256"
    num_class = 256
    train_data, test_data, _ = load_datasets("caltech_256")
    train_loaders = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loaders = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    '''

    dataset = "SUN397"
    train_loaders, val_loaders, num_class  = get_train_valid_loader("./data/SUN397/", batch_size = args.batch_size, examples_per_label=10)


    pretrained_model_dir = args.cv_dir + dataset
    if not os.path.isdir(pretrained_model_dir):
        os.mkdir(pretrained_model_dir)

    f = pretrained_model_dir + "/params.json"
    with open(f, 'wb') as fh:
        json.dump(vars(args), fh)

    results = np.zeros((4, args.nb_epochs))

    net = get_model(num_class)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        cudnn.benchmark = True

        net.cuda()   
        net = nn.DataParallel(net)

    optimizer = sgd.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr= 0.01,  momentum=0.9, weight_decay= 0.0)

    best_acc = 0.0  # best test accuracy
    start_epoch = 0

    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        adjust_learning_rate_and_learning_taks(optimizer, epoch, args)

        st_time = time.time()
        train_acc, train_loss = train(epoch, train_loaders, net, optimizer)
        print('Training Time {0}'.format(time.time()-st_time))        

        test_acc, test_loss, best_acc = test(epoch, val_loaders, net, best_acc, dataset)

        #Record statistics
        results[0:2,epoch] = [train_loss, train_acc]
        results[2:4,epoch] = [test_loss,test_acc]
        
        np.save(pretrained_model_dir + '/results', results)
