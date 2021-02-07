#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import sys

nclasses = 800*800 # Road Map Prediction

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = torch.nn.Sequential (        
            torch.nn.Linear(in_features=512,
                            out_features=nclasses),
                  torch.nn.Sigmoid())

    def forward(self, x):
        x = self.resnet(x)
        return x

