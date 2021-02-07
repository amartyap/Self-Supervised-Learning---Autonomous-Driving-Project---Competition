import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from Model import Net
from modelobj import NetObj
import numpy as np


model = Net()
mdl = NetObj()
modelObj = mdl.model

checkpoint = torch.load("modelobj/fasterrcnn_model_19.pth")
checkpoint1 = torch.load("model1/resnet18_model2_20.pth")
state_dict_1 = checkpoint1['modelRoadMap_state_dict']
state_dict_2 = checkpoint['modelObjectDetection_state_dict']
model.load_state_dict(state_dict_1)
modelObj.load_state_dict(state_dict_2)
PATH = "baseline1.pth"
torch.save({'modelRoadMap_state_dict': state_dict_1, 'modelObjectDetection_state_dict': state_dict_2},PATH)


