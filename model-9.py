#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(s):
        super().__init__()

        #hdim, zdim = 400, 64
        hdim = 128
        #s.fc1, s.fc21, s.fc22 = nn.Linear(784, hdim), nn.Linear(hdim, zdim), nn.Linear(hdim, zdim)
        #s.fc1, s.fc2= nn.Linear(784, hdim), nn.Linear(hdim, zdim)
        s.fc1= nn.Linear(784, hdim)
        #s.fc3 = nn.Linear(zdim, hdim)
        #s.fc4 = nn.Linear(hdim, 784)
        #s.fc5 = nn.Linear(zdim, 10)
        s.fc5 = nn.Linear(hdim, 10)

    #def enc(s, x):
        #x = x.view(-1,784)
        #t = F.dropout(F.relu(s.fc1(x)), 0.1, s.training)
        #t = F.relu(s.fc1(x))
        #mu, logv = s.fc21(t), s.fc22(t)
    #    std =  th.exp(0.5*logv)
    #    z = [mu + std*th.randn_like(logv) for i in range(args['nz'])]
    #    z = th.cat(z)
    #    return z, mu, logv
        #return s.fc2(t)
       # return t

    #def classify(s, z):
    #    yh = F.log_softmax(s.fc5(z), dim=1)
     #   return yh

   # def dec(s, z):
   #     xh = th.sigmoid(s.fc4(F.dropout(F.relu(s.fc3(z)), 0.1, s.training)))
   #     return xh

   # def loss(s, x, y, z, xh, mu, logv):
   #     xx = th.cat([x.view(-1,784)]*args['nz'])
  #      d = F.mse_loss(xh, xx, reduction='sum')/float(args['nz'])
  #      r = -0.5*th.sum(1+logv - mu.pow(2) - logv.exp())

  #      yh = s.classify(z)
  #      y = th.cat([y]*args['nz'])

  #      c = F.nll_loss(yh,y, reduction='sum')/float(args['nz'])
  #      yhh = yh.argmax(dim=1, keepdim=True)
  #      err = (len(yh)-yhh.eq(y.view_as(yhh)).sum().item())/float(args['nz'])

#        return d, r, c, err

    def forward(s,x):
        x = x.view(-1,784)
        #t = F.dropout(F.relu(s.fc1(x)), 0.1, s.training)
        t = F.relu(s.fc1(x))
        #z = s.enc(x)
        #xh = s.dec(z)
       # f = s.loss(x, y, z, xh, mu, logv)
        #return yh = F.log_softmax(s.fc5(z), dim=1)
        return s.fc5(t) 

