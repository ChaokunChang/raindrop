
import os
import cv2
import numpy as np
import torch
from numpy.random import RandomState
from torch.utils.data import Dataset
import os.path as op
from os.path import join as opj
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

class RainDataSet(Dataset):
    def __init__(self, raw_dir, gt_dir,mask_dir,train=True, 
                input_transform=None, target_transform=None,mask_transform=None):
        super(RainDataSet,self).__init__()
        self.raw_filenames = [opj(raw_dir, x) for x in op.listdir(raw_dir) if op.is_image_file(x)]
        self.gt_filenames = [opj(gt_dir, x) for x in op.listdir(gt_dir) if op.is_image_file(x)]
        self.mask_filenames = [opj(mask_dir, x) for x in op.listdir(mask_dir) if op.is_image_file(x)]
        assert( len(self.raw_filenames) == len(self.gt_filenames) and 
                len(self.raw_filenames) == len(self.mask_filenames) )
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        self.train = train

    def __len__(self):
        return len(self.raw_filenames)

    def __getitem__(self,index):
        input = load_img(self.raw_filenames[index])
        target = load_img(self.gt_filenames[index])
        mask = load_img(self.mask_filenames[index])
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return input,target,mask