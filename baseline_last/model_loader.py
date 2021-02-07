"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from Model import Net
from modelobj import NetObj
import numpy as np

# import your model class
# import ...

# Put your transform function here, we will use it for our dataloader
# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1():
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2():
    return torchvision.transforms.ToTensor()
    

class ModelLoader():
    # Fill the information for your team
    team_name = 'LAG'
    team_member = ["Sree Gowri Addepalli"," Amartya prasad", "Sree Lakshmi Addepalli"]
    round_number = 3
    contact_email = 'sga297@nyu.edu'

    def __init__(self, model_file="baselineFinal.pth"):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net()
        self.mdl = NetObj()
        self.modelObj = self.mdl.model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
            self.modelObj = nn.DataParallel(self.modelObj)
        checkpoint = torch.load(model_file)
        self.state_dict_1 = checkpoint['modelRoadMap_state_dict']
        self.state_dict_2 = checkpoint['modelObjectDetection_state_dict']
        self.model.load_state_dict(self.state_dict_1)
        self.modelObj.load_state_dict(self.state_dict_2)
        self.model.eval()
        self.modelObj.eval()
        self.model.to(device)
        self.modelObj.to(device)
        

    def get_bounding_boxes(self,samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        batch_size = list(samples.shape)[0]
        # Convert it into [batch_size, 3, 512, 918]
        img_tensor = self.combine_images(samples,batch_size)
        tup_boxes = []
        with torch.no_grad():
            for img in img_tensor:
              prediction = self.modelObj([img.cuda()])
              #print(prediction)
              cbox = self.convertBoundingBoxes(prediction[0]['boxes'])
              #print(cbox.shape)
              tup_boxes.append(cbox)
        return tuple(tup_boxes)

    def get_binary_road_map(self,samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        with torch.no_grad(): 
            batch_size = list(samples.shape)[0]
            sample = samples.reshape(batch_size,18,256,306)
            output = self.model(sample)
            #print(output.shape)
            output = output.reshape(800,800)
            return output


    def combine_images(self, samples, batch_size):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        ss = samples.reshape(batch_size, 2, 3, 3, 256, 306)
        t = ss.detach().cpu().clone().numpy().transpose(0, 3, 2, 1, 4, 5)
        # MergingImage
        tp = np.zeros((batch_size, 3, 3, 512, 306))
        for i in range(0, batch_size):
            for j in range(0, 3):
                for k in range(0, 3):
                    tp[i][j][k] = np.vstack([t[i][j][k][0], t[i][j][k][1]])
        tr = np.zeros((batch_size, 3, 512, 918))
        for i in range(0, batch_size):
            for j in range(0, 3):
                tr[i][j] = np.hstack([tp[i][j][0], tp[i][j][1], tp[i][j][2]])
        image_tensor = torch.from_numpy(tr).float()
        return image_tensor

    def convertBoundingBoxes(self, boxes):
        # convert [N,1,4] to [N,2,4] and handle edge cases
        if len(boxes) == 0:
            boxes = [[0,0,0,0]]
        convBoxes = []
        for box in boxes:
            xmin = box[0]
            xmin = (xmin - 400)/10
            ymin = box[1]
            ymin = (-ymin +400)/10
            xmax = box[2]
            xmax = (xmax - 400)/10
            ymax = box[3]
            ymax = (-ymax + 400)/10
            cbox = [[xmin,xmin,xmax,xmax], [ymin,ymax,ymin,ymax]]
            convBoxes.append(cbox)
        convBoxes = torch.Tensor(convBoxes)
        return convBoxes

