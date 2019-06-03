
import os
import cv2
import numpy as np
import torch
from numpy.random import RandomState
from torch.utils.data import Dataset
from torch.autograd import Variable
import os.path as op
from os.path import join as opj
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

def image_align(image,base=4):
    # IN: (HWC)
    # OUT: (CHW) the network's input is CHW
    # a_row = int(img.shape[0]/4)*4
    # a_col = int(img.shape[1]/4)*4
    # img = img[0:a_row, 0:a_col]

    image = cv2.resize(image,(224,224), interpolation=cv2.INTER_AREA)
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    # image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()
    return image

# class RainDataSet(Dataset):
#     def __init__(self, raw_dir, gt_dir,mask_dir,train=True, 
#                 input_transform=None, target_transform=None,mask_transform=None):
#         super(RainDataSet,self).__init__()
#         self.raw_filenames = [opj(raw_dir, x) for x in os.listdir(raw_dir) if is_image_file(x)]
#         self.gt_filenames = [opj(gt_dir, x) for x in os.listdir(gt_dir) if is_image_file(x)]
#         self.mask_filenames = [opj(mask_dir, x) for x in os.listdir(mask_dir) if is_image_file(x)]
#         assert( len(self.raw_filenames) == len(self.gt_filenames) and 
#                 len(self.raw_filenames) == len(self.mask_filenames) )
#         self.input_transform = input_transform
#         self.target_transform = target_transform
#         self.mask_transform = mask_transform
#         self.train = train

#     def __len__(self):
#         return len(self.raw_filenames)

#     def __getitem__(self,index):
#         input = load_img(self.raw_filenames[index])
#         target = load_img(self.gt_filenames[index])
#         mask = load_img(self.mask_filenames[index])
        
#         if self.input_transform:
#             input = self.input_transform(input)
#         if self.target_transform:
#             target = self.target_transform( target)
#         if self.mask_transform:
#             mask = self.mask_transform(mask)
#         return input,target,mask


class RainDataSet(Dataset):
    def __init__(self, raw_dir, gt_dir,mask_dir,train=True, 
                input_transform=None, target_transform=None,mask_transform=None):
        super(RainDataSet,self).__init__()
        self.raw_filenames = sorted([opj(raw_dir, x) for x in os.listdir(raw_dir) if is_image_file(x)])
        self.gt_filenames = sorted([opj(gt_dir, x) for x in os.listdir(gt_dir) if is_image_file(x)])
        self.mask_filenames = sorted([opj(mask_dir, x) for x in os.listdir(mask_dir) if is_image_file(x)])
        assert( len(self.raw_filenames) == len(self.gt_filenames) and 
                len(self.raw_filenames) == len(self.mask_filenames) )
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        self.train = train

    def __len__(self):
        return len(self.raw_filenames)

    def __getitem__(self,index):
        input = cv2.imread(self.raw_filenames[index])
        target = cv2.imread(self.gt_filenames[index])
        mask = cv2.imread(self.mask_filenames[index])
        # print(self.raw_filenames[index])
        # print(self.gt_filenames[index])
        # print(self.mask_filenames[index])
        
        input = image_align(input)
        target = image_align(target)
        mask = image_align(mask)

        return [input,target,mask]