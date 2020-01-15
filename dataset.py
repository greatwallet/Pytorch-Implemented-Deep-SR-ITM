# -*- coding: utf-8 -*-
# @Author  : Cheng Xiaotian
# @Email   : cxt16@mails.tsinghua.edu.cn

import cv2
import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np

from cv2.ximgproc import guidedFilter
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class YouTubeDataset(Dataset):
    """YouTube dataset."""

    def __init__(self, SDR_dir, HDR_dir, phase="train", scale=None, file_type='png'):
        """
        Args:
            SDR_dir (string): Directory with all the SDR images.
            HDR_dir (string): Directory with all the HDR images. 
                        Ignored if `phase` == "test" 
            phase (string): "train", "val" or "test". default: "train"
            scale (int): the scale between input and ground truth, must > 1
                        must specified for validation set and test set.
            file_type (string): File type of images. Default: png
        """
        self.SDR_dir = SDR_dir
        self.HDR_dir = HDR_dir
        self.phase = phase
        self.file_type = file_type
        self.scale = scale 
        
        if phase != "train" and phase != "val" and phase != "test":
            raise ValueError("Expecting `phase` to be {}, {} or {} but got {}"
                            .format("train", "val", "test", phase))
            
        if phase == "val" and scale is None:
            raise ValueError("scale must be specified if `phase` == {}"
                            .format(phase))
        
        N_SDR = len(glob(osp.join(self.SDR_dir,
                                  '*.{}'.format(self.file_type))))
        if self.phase != "test":
            N_HDR = len(glob(osp.join(self.HDR_dir,
                                      '*.{}'.format(self.file_type))))
            if N_SDR != N_HDR:
                raise ValueError("SDR image amount ({}) and HDR image amount({}) are not identical"
                                .format(N_SDR, N_HDR))
            
        self.len = N_SDR

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        SDR_name = osp.join(self.SDR_dir, 
                            '{:06d}.{}'.format(idx + 1, self.file_type))
        SDR_img = cv2.imread(SDR_name, cv2.IMREAD_UNCHANGED)
        
        # transfer to YUV format from UVY 
        SDR_img = cv2.cvtColor(SDR_img, cv2.COLOR_BGR2RGB)
        if self.phase == "val":
            SDR_img = cv2.resize(SDR_img, (0,0), fx=1.0/self.scale, fy=1.0/self.scale)
        
        # normalize to [0, 1]
        SDR_img = SDR_img.astype(np.float32)
        SDR_img = SDR_img / 255.0
        SDR_base = guidedFilter(guide=SDR_img, src=SDR_img, radius=5, eps=0.01)
        
        # transform to torch tensor
        SDR_img = np.moveaxis(SDR_img, -1, 0)
        SDR_base = np.moveaxis(SDR_base, -1, 0)
        SDR_img = torch.from_numpy(SDR_img)
        SDR_base = torch.from_numpy(SDR_base)
        
        # if `phase` != "test": then HDR images are provided
        if self.phase != "test":
            HDR_name = osp.join(self.HDR_dir, 
                                '{:06d}.{}'.format(idx + 1, self.file_type))
            HDR_img = cv2.imread(HDR_name, cv2.IMREAD_UNCHANGED)
            HDR_img = cv2.cvtColor(HDR_img, cv2.COLOR_BGR2RGB)
            HDR_img = HDR_img.astype(np.float32)
            HDR_img = HDR_img / 1023.0
            HDR_img = np.moveaxis(HDR_img, -1, 0)
            HDR_img = torch.from_numpy(HDR_img)
        else:
            # mark as None
            HDR_img = -1
        
        return SDR_img, HDR_img, SDR_base