#n[12]:


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys



# In[13]:


import os
import random

import numpy as np
import pandas as pd

import torchvision.transforms as transforms

import sys
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box


# In[23]:


parser = argparse.ArgumentParser(description='Road Map Prediction')


parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located.")
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 2)')
parser.add_argument('--epoch', type=int, default=75, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.09, metavar='M',
                    help='SGD momentum (default: 0.09)')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--unlabelledSceneIndex', type=int, default=106, metavar='U',
                    help='unlabelled scene index')
parser.add_argument('--labelledSceneIndex', type=int, default=120, metavar='L',
                    help='labelled scene index')
parser.add_argument('--validationSceneIndex', type=int, default=120, metavar='V',
                    help='validation scene index')
parser.add_argument('--reloadModel', type=bool, default=True, metavar='R',
                    help='LoadModel')


# In[ ]:





# In[26]:


args = parser.parse_args() # uncomment for command line


# In[8]:


random.seed(0)
np.random.seed(0)
torch.manual_seed(args.seed);


# In[ ]:


image_folder = args.data
annotation_csv = 'data/annotation.csv'


# In[ ]:


# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(args.unlabelledSceneIndex)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(args.unlabelledSceneIndex, args.labelledSceneIndex)

## validation scene index.
validation_scene_index = np.arange(args.validationSceneIndex, 134)


# In[ ]:


transform = transforms.Compose([
    transforms.ToTensor()
])


# In[ ]:


# Data Loading and validation.

# The labeled dataset can only be retrieved by sample.
# And all the returned data are tuple of tensors, since bounding boxes may have different size
# You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index,
                                  transform=transform,
                                  extra_info=True
                                 )
trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2,drop_last=True, collate_fn=collate_fn)


# In[ ]:


# The labeled dataset can only be retrieved by sample.
# And all the returned data are tuple of tensors, since bounding boxes may have different size
# You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
validation_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=validation_scene_index,
                                  transform=transform,
                                  extra_info=True
                                 )
valLoader = torch.utils.data.DataLoader(validation_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# import your model.

from Model import Net
model = Net()
print(model)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if args.reloadModel:
        model_fp = "model4/resnet18_model2_17.pth"
        model.load_state_dict(torch.load(model_fp)['modelRoadMap_state_dict'])
        print("model_loaded")


model.to(device)


# In[ ]:


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# In[ ]:


# loss func

def modify_output_for_loss_fn(loss_fn, output, dim):
    if loss_fn == "ce":
        return output
    if loss_fn == "mse":
        return F.softmax(output, dim=dim)
    if loss_fn == "nll":
        return F.log_softmax(output, dim=dim)
    if loss_fn in ["bce", "wbce", "wbce1"]:
        return torch.sigmoid(output)


# In[ ]:


# Define compute IoU

def compute_iou(pred, target):
    p = pred.cpu().numpy()
    t = target.cpu().numpy()
    I =   len(np.argwhere(np.logical_and(p==1,t==1)))/len(np.argwhere(np.logical_or(p==1,t==1)))
    U =   len(np.argwhere(np.logical_and(p==0,t==0)))/len(np.argwhere(np.logical_or(p==0,t==0)))
    mean_iou = (I+U)/2
    return mean_iou


# In[ ]:


# Define train function

def train(epoch):
    total_loss = 0
    model.train()
    model.to(device)
    for batch_idx, (sample, target, road_image, extra) in enumerate(trainloader):
        stacked_sample = torch.stack(sample)
        sampleNew = stacked_sample.reshape(args.batch_size,18,256,306)
        sampleNew = Variable(sampleNew).to(device)
        stacked_roadImage = torch.stack(road_image)
        roadImageNew = stacked_roadImage.reshape(args.batch_size, 800*800)
        roadImageNewf = roadImageNew.type(torch.float)
        sample, road_image = Variable(sampleNew).to(device), Variable(roadImageNewf).to(device)
        optimizer.zero_grad()
        output = model(sample)
        #Change loss function
        #loss = F.nll_loss(output, road_image)
        #output = F.log_softmax(output)
        loss_func = nn.BCELoss()
        loss = loss_func(output,road_image)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss_div = total_loss/ len(trainloader)
    return total_loss_div


# In[ ]:


def validation():
    model.eval()
    model.to(device)
    validation_loss = 0
    total_iou = 0
    correct = 0
    with torch.no_grad():
        for batch_idx1, (sample1, target1, road_image1, extra1) in enumerate(valLoader):
            stacked_sample1 = torch.stack(sample1)
            sampleNew1 = stacked_sample1.reshape(args.batch_size,18,256,306)
            stacked_roadImage1 = torch.stack(road_image1)
            roadImageNew1 = stacked_roadImage1.reshape(args.batch_size, 800*800)
            roadImageNewf1 = roadImageNew1.type(torch.float)
            sample1, road_image1 = Variable(sampleNew1).to(device), Variable(roadImageNewf1).to(device)
            output1 = model(sample1)
            loss_func = nn.BCELoss()
            vloss = loss_func(output1, road_image1)
            validation_loss += loss_func(output1, road_image1).data.item() # sum up batch loss
            outputNew = output1 > 0.5
            outputNew1 = outputNew.type(torch.float)
            iou = compute_iou(outputNew1,road_image1)
            total_iou += iou
        validation_loss /= len(valLoader)
        total_iou /= len(valLoader)
        return validation_loss,total_iou


# In[ ]:


for epoch in range(0, args.epoch+1):
    total_loss = train(epoch)
    print("Epoch:" + str(epoch) +" " + "total training loss: " +str(total_loss) +" \n")
    val_loss, total_iou = validation()
    print("Epoch:" + str(epoch) +" " + "total validation loss: " +str(val_loss) +" total iou: " + str(total_iou) +"\n")
    
    model_file = 'model5/resnet18_model2_' + str(epoch) + '.pth'
    torch.save({'modelRoadMap_state_dict': model.state_dict()},model_file)
    print('\nSaved model to ' + model_file )


