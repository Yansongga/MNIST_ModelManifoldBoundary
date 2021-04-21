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
    'dev': torch.device('cuda: 3' ),
    'datasize': 1000,
    'batch_size': 1000,
    'test_batch_size': 1000, 
    'epochs': 2000,
    'k_size': 1000,
    'eigen': 10,    #reducing the kernel matrix rank 
}
#saving model path
save_models_path = './models_MNIST'
check_mkdir(save_models_path)

#saving data path
save_results_path = './results_MMB'
check_mkdir(save_results_path)


trainset_all = torchvision.datasets.MNIST('./', transform=transforms.ToTensor(), download=True, train=True)
testset = torchvision.datasets.MNIST('./', transform=transforms.ToTensor(), download=True, train=False)

## train set consists of 'datasize' samples
indices = list(range(0, args['datasize']))
trainset = torch.utils.data.Subset(trainset_all, indices)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                          shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'],
                                         shuffle=False, num_workers=2)
kloader = torch.utils.data.DataLoader(trainset, batch_size=args['k_size'],
                                         shuffle=False, num_workers=2)
criterion = nn.CrossEntropyLoss()
net = FC().to(args['dev'])
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#load pre-trained model
net.load_state_dict(
    torch.load(
        os.path.join(
            save_models_path, 'FCNet={}.pth'.format( 'MNIST_1000' )            
        )))
test(args, net, testloader)
#load reduced model 
#net.load_state_dict(
#    torch.load(
#        os.path.join(
 #            save_models_path, 'eigen={}.pth'.format( args['eigen'] )            
#        )))





#model reduction by model manifold boundary
for epoch in trange(args['epochs'] ): 
    time.sleep(1)
    net.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate( kloader ):
        
        #clear gradients + add hooks + forward
        autograd_hacks.clear_backprops(net)
        optimizer.zero_grad()          
        data, target = data.to(args['dev']), target.to(args['dev'])
        autograd_hacks.add_hooks(net)
        criterion(net(data), target).backward(retain_graph=True)

        #compute per sample gradient
        autograd_hacks.compute_grad1(net)
        autograd_hacks.disable_hooks()

        #Jacobian + kernel matrix + eigen decomposition of kernel matrix 
        J =  torch.cat( 
            [params.grad1.data.view( args['k_size'], -1 )  for params in net.parameters()], 1 ) 
        kernel = torch.matmul( J, J.T )
        w, v = torch.symeig(kernel, eigenvectors=True)
        #flipping to decreasing order
        w, v = torch.flip(w, dims=[0]), torch.flip(v, dims=[1])
        
        #sample weightings for reducing kernel ranks 
        weight = v[:, args['eigen'] - 1 ] ** 2
        if epoch == 0:
            w0, v0= w + 0., v + 0.

        #weighted gradient + optimize + clear grad1
        weighted_loss_grad(net, weight)
        optimizer.step()
        zero_grad1(net)
     
    #print results 
    if (epoch + 1) % 5 == 0 or epoch == 0: 
        test(args, net, testloader)
        print('\nEigenvalue: {:.4f}. \n'.format( w[args['eigen'] - 1 ] ))



## save reduced model 
#torch.save(
#    net.state_dict(), 
#                   os.path.join(save_models_path, 
#                               'eigen={}.pth'.format( args['eigen'] )  
#                               )
#)




#plot log eigenvalue 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

num = np.array([k+1 for k in range(0,args['datasize'])])
sqr = w[0:args['datasize']].log().cpu().numpy()
sqr2 = w0[0:args['datasize']].log().cpu().numpy()

# convert to pandas dataframe
d = {'eigen order': num, 'log MMB eigen value': sqr }
d2 = {'eigen order': num, 'log eigen value': sqr2}
pdnumsqr = pd.DataFrame(d)
pdnumsqr2 = pd.DataFrame(d2)

sns.set(style='darkgrid')
sns.lineplot(x='eigen order', y='log MMB eigen value', data=pdnumsqr)
sns.lineplot(x='eigen order', y='log eigen value', data=pdnumsqr2)


#pre-train model
#for epoch in range(args['epochs']):  # loop over the dataset multiple times
#    train_epoch( args, net, optimizer, trainloader )
 #   if (epoch + 1) % 100 == 0:
#        test(args, net, testloader)
 
#print('Finished Training')

####save pretrained model
#torch.save(
 #   net.state_dict(), 
#                   os.path.join(save_models_path, 
#                               'FCNet={}.pth'.format( 'MNIST' )  
#                               )
#)
