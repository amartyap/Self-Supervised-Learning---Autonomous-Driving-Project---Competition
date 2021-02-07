import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torchvision.transforms as transforms
from torch import Tensor

from helper import convert_map_to_lane_map, convert_map_to_road_map

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
]


# The dataset class for unlabeled data.
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform

        assert first_dim in ['sample', 'image']
        self.first_dim = first_dim

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                images.append(self.transform(image))
            image_tensor = torch.stack(images)

            return image_tensor

        elif self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name)

            image = Image.open(image_path)

            return self.transform(image), index % NUM_IMAGE_PER_SAMPLE


# The dataset class for labeled data.
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=True):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images)

        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()

        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)

        target = {}
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        target['category'] = torch.as_tensor(categories)

        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)

            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return image_tensor, target, road_image, extra

        else:
            return image_tensor, target, road_image


# The dataset class for unlabeled data.
class SimclrUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform
        # self.labels = np.asarray([-1] * self.data.shape[0])

        assert first_dim in ['sample', 'image']
        self.first_dim = first_dim

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                p, r = self.transform(image)
                images.append(p)
                images.append(r)
            image_tensor = torch.stack(images)
            # print(image_tensor.shape)
            image_tensor = image_tensor.reshape(-1, 18, 256, 306)

            return image_tensor, int(-1)

        elif self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name)

            image = Image.open(image_path)

            return self.transform(image), index % NUM_IMAGE_PER_SAMPLE


# The dataset class for labeled data for RoadMap.
class SimclrLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=False):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images)
        image_tensor = image_tensor.reshape(18, 256, 306)

        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()

        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)

        target = {}
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        target['category'] = torch.as_tensor(categories)

        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)

            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return image_tensor, target, road_image, extra

        else:
            return image_tensor, road_image


class SimclrLabeledBBDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=False):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def reshapeTarget(self, target):
        targetNew = []
        for j in target:
            # print("!!!")
            # print(j)
            # print("!!!")
            rescaled_x_coordinates = j[0] * 10 + 400
            rescaled_y_coordinates = -j[1] * 10 + 400
            # print("rescaled")
            # print(rescaled_x_coordinates)
            # print(rescaled_y_coordinates)
            # print("###")
            xmin = torch.min(rescaled_x_coordinates)
            ymin = torch.min(rescaled_y_coordinates)
            xmax = torch.max(rescaled_x_coordinates)
            ymax = torch.max(rescaled_y_coordinates)
            bbd = torch.as_tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
            torch.set_printoptions(precision=10)
            # print(bbd.shape)
            targetNew.append(bbd)
        # tr = torch.Tensor(targetNew)
        # print(len(targetNew))
        tr = torch.stack(targetNew)
        #print("$$$$$")
        #print(tr.shape)
        # print("%%%%")
        # print(tr)
        # print("%%%%")
        return tr

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images)
        image_tensor = image_tensor.reshape(18, 256, 306)

        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()

        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)

        target = {}
        bboxes = torch.as_tensor(corners).view(-1, 2, 4)
        #print("***")
        #print(bboxes.shape)
        target['boxes'] = self.reshapeTarget(bboxes)
        target['category'] = torch.as_tensor(categories)

        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)

            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return image_tensor, target, road_image, extra

        else:
            # print(type(image_tensor))
            # print(type(target))
            return image_tensor, target


# Dataset for updated labelled dataset for object detection.

class ObjDetectionLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=True, batch_size=2):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
        self.batch_size = batch_size
        # self.reshapeTarget = reshapeTarget

    def reshapeTarget(self, target):
        targetNew = []
        for j in target:
            # print("!!!")
            # print(j)
            # print("!!!")
            rescaled_x_coordinates = j[0] * 10 + 400
            rescaled_y_coordinates = -j[1] * 10 + 400
            # print("rescaled")
            # print(rescaled_x_coordinates)
            # print(rescaled_y_coordinates)
            # print("###")
            xmin = torch.min(rescaled_x_coordinates)
            ymin = torch.min(rescaled_y_coordinates)
            xmax = torch.max(rescaled_x_coordinates)
            ymax = torch.max(rescaled_y_coordinates)
            bbd = torch.as_tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
            torch.set_printoptions(precision=10)
            targetNew.append(bbd)
        tr = torch.stack(targetNew)
        # print("%%%%")
        # print(tr)
        # print("%%%%")
        return tr

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images)
        # Combining all the six sample images into one.
        # stacked_sample = torch.stack(image_tensor)
        '''
        ss = image_tensor.reshape(2, 3, 3, 256, 306)
        t = ss.numpy().transpose(2, 1, 0, 3, 4)
        # MergingImage
        tp = np.zeros((3, 3, 512, 306))
        for j in range(0, 3):
            for k in range(0, 3):
                tp[j][k] = np.vstack([t[j][k][0], t[j][k][1]])
        # print("****")
        # print(tp.shape)
        tr = np.zeros((3, 512, 918))
        for j in range(0, 3):
            tr[j] = np.hstack([tp[j][0], tp[j][1], tp[j][2]])
        # print("####")
        # print(tr.shape)
        #td = np.zeros((3, 800, 800))
        #for j in range(0, 3):
            #td[j] = cv2.resize(tr[j], dsize=(800, 800))
        # print(td)
        sampleNew = torch.from_numpy(tr).float()
        sampleNew = transforms.ToPILImage()(sampleNew.squeeze_(0)).convert("RGB")
        image_tensor = sampleNew'''
        image_tensor = image_tensor.reshape(-1, 18, 256, 306)
        # print(image_tensor.shape)
        sampleNew = transforms.ToPILImage()(image_tensor.squeeze_(0)).convert("RGB")
        image_tensor = sampleNew

        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()

        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)

        target = {}
        # target["image_id"] = Tensor([sample_id], dtype = torch.int64)
        t = torch.zeros(1, dtype=torch.int64)
        t[0] = sample_id
        target["image_id"] = t
        # print("*****")
        # print("image_id type:")
        # print(target["image_id"].type())
        # print(target["image_id"].dtype)
        bboxes = torch.as_tensor(corners, dtype=torch.float32).view(-1, 2, 4)
        # print(bboxes)
        target["boxes"] = self.reshapeTarget(bboxes)
        # print("box type:")
        # print(target["boxes"].type())
        # print(target["boxes"].dtype)
        target["labels"] = torch.as_tensor(categories, dtype=torch.int64)
        # print("labels type:")
        # print(target["labels"].type())
        # print(target["labels"].dtype)
        boxes = target["boxes"]
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # print(area)
        # print("area type:")
        target["area"] = area
        # print(target["area"].type())
        # print(target["area"].dtype)
        # print("iscrowd type:")
        target["iscrowd"] = torch.zeros(len(target["boxes"]), dtype=torch.int64)
        target["GTRoadImage"] = road_image
        # print(target["iscrowd"].type())
        # print(target["iscrowd"].dtype)
        # print("img type:")
        # print(type(image_tensor))
        # print(image_tensor.dtype)
        # print("*****")
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return image_tensor, target    