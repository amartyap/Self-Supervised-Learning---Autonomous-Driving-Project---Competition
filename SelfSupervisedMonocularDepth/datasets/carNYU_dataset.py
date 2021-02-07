# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# Designing class for our dataset for monocular depth estimation.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

import random
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class carNYUDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(carNYUDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[879.03824732 / 1224, 0, 613.17597314 / 1224, 0],
                           [0, 879.03824732 / 1024, 524.14407205 / 1024, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (306, 256)
        # self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def get_image_path(self, folder, scene_index, samp_index, image, side):
        #print(samp_index)
        f_str = f"{folder}/scene_{scene_index}/sample_{samp_index}/{image}"
        image_path = os.path.join(self.data_path, f_str)
        return image_path

    def check_depth(self):
        return False

    def get_color(self, folder, scene_index, sample_index, image, side, do_flip):
        color = self.loader(self.get_image_path(folder, scene_index, sample_index, image, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        self.path = self.filenames[index]
        line = self.filenames[index].split("/")
        #print(line)
        #print("!!!!")

        folder = line[0] + "/" + line[1] + "/" + line[2] + "/" + line[3]
        scene = line[4]
        scene_index = int(scene.split("_")[1])
        #print(scene_index)
        sample = line[5]
        sample_index = int(sample.split("_")[1])
        b = sample_index
        #sample_index = 5
        #print(sample_index)
        image = line[6]
        side = 4

        if sample_index < 1:
          sample_index = 1
        if sample_index > 124:
          sample_index = 124

        

        for i in self.frame_idxs:
          #print(sample_index)
          #print(i)
          inputs[("color", i, -1)] = self.get_color(folder, scene_index, sample_index+i, image, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        return inputs