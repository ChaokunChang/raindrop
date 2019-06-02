#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataloader
from torch import optim
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import sys
import argparse
from os.path import join as opj

from tensorboardX import SummaryWriter

#Models lib
from models import *
#dataset lib
from dataset import RainDataSet
#Metrics lib
from metrics import calc_psnr, calc_ssim

from pytorch.models import Discriminator,Generator
from pytorch.dataset import RainDataSet
from pytorch.metrics import calc_psnr,calc_ssim

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action='store_true',help='train model')
    parser.add_argument("--predict",action='store_true',help='predict the result')
    parser.add_argument("--gpu",type=str,default='0',help='specify gou devices')
    parser.add_argument("--mode", type=str,default='demo',choices=['demo','test','continue'])

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='Adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--interval', type=float, default=20,
                                help='the interval of l_r')
    train_settings.add_argument('--ratio', type=float, default=20,
                                help='the ratio of losses.')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')
    train_settings.add_argument('--show_every_nsteps', type=int, default=10,
                                help='calculate metrics(RMSE,PSNR) every n steps.')
    train_settings.add_argument('--evaluate_every', type=int, default=10,
                                help='evaluate the model every n epochs.')

    path_settings = parser.add_argument_group('path settings')
    parser.add_argument("--input_dir", type=str,default="../data/demo/input/raw")
    parser.add_argument("--gt_dir", type=str,default="../data/demo/input/gt")
    parser.add_argument("--output_dir", type=str,default="../data/demo/output")
    parser.add_argument("model_dir",type=str,default="./weights",
                        help= "the dir to store the model(weights) after train progress.")
    parser.add_argument("--g_weights",type=str,default="gen.pkl",
                        help='the file of weights to load in train progress.')
    parser.add_argument("--d_weights",type=str,default="dis.pkl",
                        help='the file of weights to load in train progress.')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path', default='../logs/',
                               help='path of the log file. If not set, logs are printed to console')

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

def train_epoch(model:tuple, optimizer:tuple, dataloader:Dataloader,epoch, criterion:tuple, 
                metric:tuple,args,writer=None):
    """ Train a single epoch.
    :param model
    :param optimizer
    :param dataloader
    :param loss
    :param metric
    :param writer
    :param loss_ratio
    :return None
    """
    g_model,d_model = model
    g_optimizer,d_optimizer = optimizer
    g_criterion,d_criterion = criterion
    ssim_metric,psnr_metric = metric

    data_num = len(dataloader)
    # First we train the generative model.
    # and then we train the discrimnative model
    g_model.train()
    d_model.train()
    g_loss_list=[]
    d_loss_list=[]
    for i,data in enumerate(dataloader):
        g_model.zero_grad()
        g_optimizer.zero_grad()
        input_data,mask_data,gt_data = data[0],data[1],data[2]
        mask_list, skip1, skip2, g_output = g_model.forward(input_data)
        g_loss,attentive_rnn_loss,autoencoder_loss = g_criterion(mask_list,mask_data,gt_data) #generate two loss
        g_loss.backward()
        g_optimizer.step()
        g_loss_list.append(g_loss)

        d_model.zero_grad()
        d_optimizer.zero_grad()
        mask, d_output = d_model(g_output)
        d_loss,map_loss = d_criterion(mask,mask_list[-1],gt_data) #here we will call forward with gt_data
        d_loss.backward()
        d_optimizer.step()
        d_loss_list.append(d_loss)

        if i % args.show_every_nsteps == 0:
            with torch.no_grad():
                ssim_loss = ssim_metric(gt_data,g_output)
                psnr_loss = psnr_metric(gt_data,g_output)
                print('epoch {}, [{}/{}], loss ({},{}), PSNR {}, SSIM {},'.format
                (epoch, i, data_num, g_loss,d_loss, psnr_loss, ssim_loss))
                if writer is not None:
                    step = epoch * data_num + i
                    writer.add_scalar('G_Loss', g_loss.item(), step)
                    writer.add_scalar('D_Loss', d_loss.item(), step)
                    writer.add_scalar('SSIM', ssim_loss.item(), step)
                    writer.add_scalar('PSNR', psnr_loss.item(), step)
    return (np.mean(g_loss_list),np.mean(d_loss_list)),(ssim_loss,psnr_loss)


def train(args):
    generate_model = Generator().cuda()
    discriminate_model = Discriminator.cuda()
    if args.mode == 'continue':
        generate_model = generate_model.load_state_dict(torch.load(opj(args.model_dir,args.g_weights)) )
        discriminate_model = discriminate_model.load_state_dict(torch.load(opj(args.model_dir,args.d_weights)))
    model = (generate_model,discriminate_model)

    data = RainDataSet(args.input_dir)
    train_data = data.get_train()
    test_data = data.get_test()
    train_loader = Dataloader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=0)
    test_loader = Dataloader(test_data,batch_size=args.batch_size,shuffle=False,num_workers=0)

    beta1 = 0.9
    beta2 = 0.999
    G_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate, weight_decay=args.weight_decay, betas=(beta1, beta2))
    D_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate, weight_decay=args.weight_decay, betas=(beta1, beta2))
    optimizer = (G_optimizer, D_optimizer)

    G_criterion = GeneratorLoss()
    D_criterion = DiscriminatorLoss()
    criterion = (G_criterion,D_criterion)

    ssim_criterion = SSIM()
    psnr_criterion = PSNR()
    metric = (ssim_criterion,psnr_criterion)

    summary_file =  'DRNet_{}_{}_{}_{}_{}_{}'.format(args.learning_rate, args.weight_decay,
                    args.radio,args.batch_size,args.interval)
    summary_path = opj(args.summary_dir,summary_file)
    print("The summary is stored in {}".format(summary_path))
    writer = SummaryWriter(summary_path)

    for epoch in args.epochs:
        start_time = time.time()
        metric[0].reset()
        metric[1].reset()
        current_lr = args.learning_rate / 2**int(epoch/args.interval)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print("Train_epoch_{0}: learning_rate= {1}".format(epoch,current_lr))
        loss,acc = train_epoch(model,optimizer,train_loader,epoch,criterion,metric,args,writer=writer)
        print(' Train_epoch_{0} : G_Loss= {:.5f}; D_Loss= {:.5f}; '
                'SSIM= {:.5f}; PSNR= {:.5f}'.format(epoch,loss[0].item(),
                loss[1].item(),acc[0].item(),acc[1].item()))
        if epoch % args.evaluate_every:
            evaluate()


def evaluate():
    print("Not Implement Error.")
    pass


def predict_single(model,image):
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()
    out = model(image)[-1]
    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :]*255.
    return out


def predict(args):
    model = Generator().cuda()
    model.load_state_dict(torch.load(opj(args.model_dir, args.g_weights)))

    if args.mode == 'demo':
        input_list = sorted(os.listdir(args.input_dir))
        num = len(input_list)
        for i in range(num):
            print ('Processing image: %s'%(input_list[i]))
            img = cv2.imread(opj(args.input_dir , input_list[i]))
            img = image_align(img)
            result = predict_single(model,img)
            img_name = input_list[i].split('.')[0]
            cv2.imwrite(opj(args.output_dir, img_name + '.jpg'), result)

    elif args.mode == 'test':
        input_list = sorted(os.listdir(args.input_dir))
        gt_list = sorted(os.listdir(args.gt_dir))
        num = len(input_list)
        cumulative_psnr = 0
        cumulative_ssim = 0
        for i in range(num):
            print ('Processing image: %s'%(input_list[i]))
            img = cv2.imread(opj(args.input_dir , input_list[i]))
            gt = cv2.imread(opj(args.gt_dir , gt_list[i]))
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