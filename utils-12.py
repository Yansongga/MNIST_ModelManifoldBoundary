import os, pdb, sys, json, subprocess,        time, logging, argparse,        pickle, math, gzip, numpy as np,        glob
       #pandas as pd,        
from functools import partial, reduce
from pprint import pprint
from copy import deepcopy

import  torch as th, torch.nn as nn,         torch.backends.cudnn as cudnn,         torchvision as thv,         torch.nn.functional as F,         torch.optim as optim
from torchvision import transforms
cudnn.benchmark = True
import torch

from collections import defaultdict
from torch._six import container_abcs
import torch
from copy import deepcopy
from itertools import chain
from model import *
from torch.utils.data import DataLoader
import math

import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import random
import torchvision.models as models_t
import matplotlib.pyplot as plt  
import torch.optim as optim

from tqdm import trange


import time

# save model
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

#train
def train_epoch(args, network, optimizer, dl):
    network.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate( dl ):
        optimizer.zero_grad()
        data, target = data.to(args['dev']), target.to(args['dev'])
        output =  network(data)
        #loss = criterion(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

#test function
def test(args, network, testloader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(args['dev']), target.to(args['dev'])
            output = F.log_softmax(  network(data) )
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(testloader.dataset), 
        100. * correct / len(testloader.dataset)))
  

#### zero per sample gradient
def zero_grad1(net):
    for p in net.parameters():
        p.grad1.zero_()

###
def weighted_loss_grad(net, weight):
    for p in net.parameters():
        if p.data.dim() == 2:
            p.grad.data = torch.einsum('i, ijk -> jk', weight, (p.grad1.data + 0.) ) + 0.
        if p.data.dim() == 1:
            p.grad.data = torch.einsum('i, ij -> j', weight, (p.grad1.data + 0.) ) + 0.