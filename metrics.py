import skimage
import cv2
from skimage.measure import compare_psnr, compare_ssim
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchnet import meter
import numpy as np

def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)

def psnr_ssim_metric(images1,images2):
    batch_size = images1.size()[0]
    images1 = images1.cpu().data
    images2 = images2.cpu().data
    cumulative_psnr=0
    cumulative_ssim=0
    for i in range(batch_size):
        img1 = images1[i]
        img2 = images2[i]
        img1 = np.array(img1, dtype = 'uint8').transpose((1, 2, 0))
        img2 = np.array(img2, dtype = 'uint8').transpose((1, 2, 0))
        cur_psnr = calc_psnr(img1,img2)
        cur_ssim = calc_ssim(img1,img2)
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
    return cumulative_psnr/batch_size, cumulative_ssim/batch_size