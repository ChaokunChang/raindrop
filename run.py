#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
#Models lib
from models import *
#Metrics lib
from metrics import calc_psnr, calc_ssim


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action='store_true',help='train model')
    parser.add_argument("--predict",action='store_true',help='predict the result')
    parser.add_argument("--gpu",type=str,default='0',help='specify gou devices')

    parser.add_argument("--mode", type=str,default='demo',choices=['demo','test','continue'])
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--weights_path",type=str,default="./weights/gen.pkl",
                        help='the path of weights to load in train progress.')
    parser.add_argument("model_save_path",type=str,default="./weights",
                        help= "the path to store the model(weights) after train progress.")

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='None',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    args = parser.parse_args()
    return args

def image_align(img,base=4):
    #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]
    #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


def train(args):
    model = Generator().cuda()
    if args.mode == 'continue':
        model.load_state_dict(torch.load(args.weights_path))
    


def predict(args):
    model = Generator().cuda()
    model.load_state_dict(torch.load(args.weights_path))

    if args.mode == 'demo':
        input_list = sorted(os.listdir(args.input_dir))
        num = len(input_list)
        for i in range(num):
            print ('Processing image: %s'%(input_list[i]))
            img = cv2.imread(os.path.join(args.input_dir , input_list[i]))
            img = image_align(img)
            # result = predict(img)
            image = np.array(img, dtype='float32')/255.
            image = image.transpose((2, 0, 1))
            image = image[np.newaxis, :, :, :]
            image = torch.from_numpy(image)
            image = Variable(image).cuda()

            out = model(image)[-1]

            out = out.cpu().data
            out = out.numpy()
            out = out.transpose((0, 2, 3, 1))
            out = out[0, :, :, :]*255.
            result = out

            img_name = input_list[i].split('.')[0]
            cv2.imwrite(os.path.join(args.output_dir, img_name + '.jpg'), result)

    elif args.mode == 'test':
        input_list = sorted(os.listdir(args.input_dir))
        gt_list = sorted(os.listdir(args.gt_dir))
        num = len(input_list)
        cumulative_psnr = 0
        cumulative_ssim = 0
        for i in range(num):
            print ('Processing image: %s'%(input_list[i]))
            img = cv2.imread(os.path.join(args.input_dir , input_list[i]))
            gt = cv2.imread(os.path.join(args.gt_dir , gt_list[i]))
            img = image_align(img)
            gt = image_align(gt)
            result = predict(img)
            result = np.array(result, dtype = 'uint8')
            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)
            print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
        print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/num, cumulative_ssim/num))

    else:
        print ('Mode Invalid!')
    pass


def run():
    args = get_args()

    if args.train:
        train(args)
    elif args.predict:
        predict(args)
    else:
        print("You won't run.")