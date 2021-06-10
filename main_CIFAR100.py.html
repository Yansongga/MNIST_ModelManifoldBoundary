#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
from model import *
import torch.optim as optim
import autograd_hacks
from utils import * 
import torch.nn.functional as F
from itertools import chain
from tqdm import trange
import time


args = {
    'task': 'herbivores', 
    'herbivores': [15, 19, 21, 31, 38],  
    'carnivores': [3, 42, 43, 88, 97], 
    'vehicles-1':  [8, 13, 48, 58, 90], 
    'vehicles-2': [ 41, 69, 81, 85, 89], 
    'flowers': [ 54, 70, 62, 82, 92 ], 
    'scenes': [ 23, 33, 49, 60, 71 ], 
    'dev': torch.device('cuda' ),
    'datasize': 2500,
   # 'batch_size': 100,
    'test_batch_size': 500, 
    'epochs': 10000,
    'k_size': 1000,
    'eigen': 20,    #reducing the kernel matrix rank 
}
args['batch_size'] = args['datasize']
#saving model path
save_models_path = './models_CIFAR'
check_mkdir(save_models_path)

#saving data path
save_results_path = './results_CIFAR'
check_mkdir(save_results_path)

#transformation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])
#loading data
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train= False, download=True, transform=transform_test)

#split doamin
trainset, testset = data_split( trainset, args[ 'herbivores' ], 0 ), data_split( testset, args[ 'herbivores'], 0 )

## train set consists of 'datasize' samples
#indices = list(range(0, args['datasize']))
#trainset = torch.utils.data.Subset(trainset_all, indices)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['datasize'],
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'],
                                         shuffle=False, num_workers=2)
#kloader = torch.utils.data.DataLoader(trainset, batch_size=args['k_size'],
#                                         shuffle=False, num_workers=2)
criterion = nn.CrossEntropyLoss()


# In[2]:


net = CNN().to(args['dev'])
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-4)
optimizer = optim.SGD(net.parameters(), lr=0.001 )


# In[3]:


args['eigen']


# In[ ]:


#model reduction by model manifold boundary
for batch_idx, (data, target) in enumerate( trainloader ):
     data, target = data.to(args['dev']), target.to(args['dev'])
for epoch in trange( 100000 ): 
    time.sleep(1)
    net.train()
    criterion = nn.CrossEntropyLoss()
    
        
    #clear gradients + add hooks + forward
    autograd_hacks.clear_backprops(net)
    optimizer.zero_grad()          
    autograd_hacks.add_hooks(net)
    criterion(net(data), target).backward(retain_graph=True)

    #compute per sample gradient
    autograd_hacks.compute_grad1(net)
    autograd_hacks.disable_hooks()

    #Jacobian + kernel matrix + eigen decomposition of kernel matrix 
    J =  torch.cat( 
        [params.grad1.data.view( args['batch_size'], -1 )  for params in net.parameters()], 1 ) 
    kernel = torch.matmul( J, J.T )
    w, v = torch.symeig(kernel, eigenvectors=True)
    #flipping to decreasing order
    w, v = torch.flip(w, dims=[0]), torch.flip(v, dims=[1])

    #sample weightings for reducing kernel ranks 
    weight = v[:, args['eigen'] - 1 ] ** 2
    #balance sample trades 
    weight = 0.9 * weight + 0.1  / data.size(0)
    #print(weight.sum())
    if epoch == 0:
        w0, v0= w + 0., v + 0.

    #weighted gradient + optimize + clear grad1
    weighted_loss_grad(net, weight)
    optimizer.step()
    zero_grad1(net)

    pa =  torch.cat( 
        [params.data.view( -1 )  for params in net.parameters()], 0 ) 
    n_p = torch.norm(pa)
    if n_p > 12.:
        for p in net.parameters():
            p.data /= ( n_p /12.0 )
    #autograd_hacks.clear_backprops(net)
     
    #print results 
    if (epoch + 1) % 100 == 0 or epoch == 0: 
        test(args, net, testloader)
        print('\nEigenvalue: {:.4f}. \n'.format( w[args['eigen'] - 1 ] ))
        print('\nNorm: {:.4f}. \n'.format( n_p.item() ))


# In[ ]:




