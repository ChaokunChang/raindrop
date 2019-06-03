#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
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
import pandas as pd
import pickle as pkl

from tensorboardX import SummaryWriter

#Models lib
from models import *
#dataset lib
from dataset import RainDataSet
#Metrics lib
from metrics import calc_psnr, calc_ssim,psnr_ssim_metric
# from metrics SSIM,PSNR
#Losses lib
from losses import GeneratorLoss,DiscriminatorLoss
# Tranforms lib
from utils.transforms import demo_transform1,demo_transform2,image_align
from utils.funcs import status
# from pytorch.models import Discriminator,Generator
# from pytorch.dataset import RainDataSet
# from pytorch.metrics import calc_psnr, calc_ssim, SSIM, PSNR
# from pytorch.losses import GeneratorLoss,DiscriminatorLoss
# from pytorch.utils.transforms import demo_transform1,demo_transform2,image_align

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action='store_true',help='train model')
    parser.add_argument("--predict",action='store_true',help='predict the result')
    parser.add_argument("--split",action='store_true',help='split the result')
    parser.add_argument("--gpu",type=str,default='0',help='specify gou devices')
    parser.add_argument("--mode", type=str,default='demo',choices=['demo','test','continue'])
    parser.add_argument("--prepare",action='store_true',help='prepare the mask')

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
    train_settings.add_argument('--batch_size', type=int, default=1,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')
    train_settings.add_argument('--show_every_nsteps', type=int, default=10,
                                help='calculate metrics(RMSE,PSNR) every n steps.')
    train_settings.add_argument('--evaluate_every', type=int, default=10,
                                help='evaluate the model every n epochs.')
    train_settings.add_argument('--save_every', type=int, default=5,
                                help='save the model every n epochs.')

    path_settings = parser.add_argument_group('path settings')
    parser.add_argument("--input_dir", type=str,default="../data/demo/train/raw")
    parser.add_argument("--gt_dir", type=str,default="../data/demo/train/gt")
    parser.add_argument("--mask_dir", type=str,default="../data/demo/train/mask")
    parser.add_argument("--output_dir", type=str,default="../data/demo/train/output")

    parser.add_argument("--test_input_dir", type=str,default="../data/raindrop_data/test_a/raw")
    parser.add_argument("--test_gt_dir", type=str,default="../data/raindrop_data/test_a/gt")
    parser.add_argument("--test_mask_dir", type=str,default="../data/raindrop_data/test_a/mask")
    parser.add_argument("--test_output_dir", type=str,default="../data/raindrop_data/test_a/output")
    
    parser.add_argument("--split_dir", type=str,default="../data/ssim_split")
    parser.add_argument("--model_dir",type=str, default="weights",
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

def train_epoch(model:tuple, optimizer:tuple, dataloader:DataLoader,epoch, criterion:tuple, 
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
    # ssim_metric,psnr_metric = metric

    data_num = len(dataloader)
    # First we train the generative model.
    # and then we train the discrimnative model
    g_loss_list=[]
    d_loss_list=[]
    psnr_loss = 0
    ssim_loss = 0
    for i,data in enumerate(dataloader):
        g_model.train()
        g_model.zero_grad()
        g_optimizer.zero_grad()
        
        # input_data,gt_data,mask_data = torch.squeeze(data[0],0),torch.squeeze(data[1],0),torch.squeeze(data[2],0)
        input_data,gt_data,mask_data = data[0],data[1],data[2]
        mask_list, skip1, skip2, g_output = g_model.forward(input_data)
        # print("g_output:",g_output.size())
        g_loss,interlosses = g_criterion(mask_list,mask_data,gt_data,
                                        [skip1, skip2, g_output]) #generate two loss
        #g_loss.backward(retain_graph=True)
        g_loss.backward()        
        g_optimizer.step()
        g_loss_list.append(g_loss)

        d_model.train()
        d_model.zero_grad()
        d_optimizer.zero_grad()
        mask1, d_output1 = d_model(g_output.detach_())
        mask2, d_output2 = d_model(gt_data)
        d_loss,map_loss = d_criterion(mask1,d_output1,mask2,d_output2,mask_list[-1].detach_()) #here we will call forward with gt_data
        d_loss.backward()

        d_optimizer.step()
        d_loss_list.append(d_loss)
        torch.cuda.empty_cache()
        if i % args.show_every_nsteps == 0:
            with torch.no_grad():
                # print(gt_data.size())
                # print(g_output.size())
                
                psnr_loss,ssim_loss = metric(gt_data,g_output.detach_())
                print('epoch {}, [{}/{}], loss ({},{}), PSNR {}, SSIM {},'.format
                (epoch, i, data_num, g_loss.data,d_loss.data, psnr_loss, ssim_loss))
                if writer is not None:
                    step = epoch * data_num + i
                    writer.add_scalar('G_Loss', g_loss.item(), step)
                    writer.add_scalar('D_Loss', d_loss.item(), step)
                    writer.add_scalar('SSIM', ssim_loss.item(), step)
                    writer.add_scalar('PSNR', psnr_loss.item(), step)
    r_g_loss = torch.mean(torch.stack(g_loss_list))
    r_d_loss = torch.mean(torch.stack(d_loss_list))
    return r_g_loss,r_d_loss,ssim_loss,psnr_loss


def train(args):
    generate_model = Generator().cuda()
    discriminate_model = Discriminator().cuda()
    if args.mode == 'continue':
        generate_model = generate_model.load_state_dict(torch.load(opj(args.model_dir,args.g_weights)) )
        discriminate_model = discriminate_model.load_state_dict(torch.load(opj(args.model_dir,args.d_weights)))
    model = (generate_model,discriminate_model)

    train_data = RainDataSet(args.input_dir,args.gt_dir,args.mask_dir)
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=0)
    # test_data = data.get_test()
    # test_loader = DataLoader(test_data,batch_size=args.batch_size,shuffle=False,num_workers=0)

    beta1 = 0.9
    beta2 = 0.999
    G_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model[0].parameters()),
                            lr=args.learning_rate, weight_decay=args.weight_decay, betas=(beta1, beta2))
    D_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model[1].parameters()),
                            lr=args.learning_rate, weight_decay=args.weight_decay, betas=(beta1, beta2))
    optimizer = (G_optimizer, D_optimizer)

    G_criterion = GeneratorLoss()
    D_criterion = DiscriminatorLoss()
    criterion = (G_criterion,D_criterion)

    metric = psnr_ssim_metric

    summary_file = 'DRNet_{}_{}_{}_{}_{}'.format(args.learning_rate, args.weight_decay,
                    args.batch_size,args.ratio,args.interval)
    summary_path = opj(args.summary_dir,summary_file)
    print("The summary is stored in {}".format(summary_path))
    writer = SummaryWriter(summary_path)
    best_ssim=0
    best_psnr=0
    for epoch in range(args.epochs):
        start_time = time.time()
        # metric[0].reset()
        # metric[1].reset()
        current_lr = args.learning_rate / 2**int(epoch/args.interval)
        for param_group in optimizer[0].param_groups:
            param_group["lr"] = current_lr
        for param_group in optimizer[1].param_groups:
            param_group["lr"] = current_lr
        print("Train_epoch_{0}: learning_rate= {1}".format(epoch,current_lr))
        loss1,loss2,acc1,acc2 = train_epoch(model,optimizer,train_loader,epoch,criterion,metric,args,writer=writer)
        print("Train_epoch_{:d} : G_Loss= {:.5f}; D_Loss= {:.5f}; SSIM= {:.5f}; PSNR= {:.5f}".format(epoch,loss1.data,loss2.data,acc1,acc2))
        if epoch % args.evaluate_every:
            ssim,psnr = evaluate(generate_model,args)
            if (ssim > best_ssim) or (ssim == best_ssim and psnr>best_psnr):
                best_ssim = ssim
                best_psnr = psnr
                print("Get better model.")
                g_path = opj(args.model_dir,"gen_best")
                d_path = opj(args.model_dir,"dis_best")
                torch.save(generate_model.state_dict(),g_path)
                torch.save(discriminate_model.state_dict(),d_path)
                
            pass
        if epoch % args.save_every:
            g_path = opj(args.model_dir,"gen_{}".format(epoch))
            torch.save(generate_model.state_dict(),g_path)
            d_path = opj(args.model_dir,"dis_{}".format(epoch))
            torch.save(discriminate_model.state_dict(),d_path)
            print("Model saved in epoch_{}".format(epoch))
        
def evaluate(model,args):
    input_list = sorted(os.listdir(args.test_input_dir))
    gt_list = sorted(os.listdir(args.test_gt_dir))
    num = len(input_list)
    cumulative_psnr = 0
    cumulative_ssim = 0
    psnr_list = []
    ssim_list = []
    for i in range(num):
        prefix = input_list[i].split('_')[0]
        print ('Processing image: %s'%(input_list[i]))
        img = cv2.imread(opj(args.test_input_dir , input_list[i]))
        gt = cv2.imread(opj(args.test_gt_dir , gt_list[i]))
        img = image_align(img)
        gt = image_align(gt)
        result = predict_single(model,img)
        result = np.array(result, dtype = 'uint8')
        cur_psnr = calc_psnr(result, gt)
        cur_ssim = calc_ssim(result, gt)
        print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
        psnr_list.append(cur_psnr)
        ssim_list.append(cur_ssim)
        out_name = prefix+"_"+"output.png"
        cv2.imwrite( opj( args.test_output_dir,out_name), result )
    print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/num, cumulative_ssim/num))
    df = pd.DataFrame(np.array([psnr_list,ssim_list]).T, columns=['psnr','ssim'])
    df.head()
    print(df.apply(status))
    return np.mean(ssim_list) , np.mean(psnr_list)


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
        psnr_list = []
        ssim_list = []
        for i in range(num):
            print ('Processing image: %s'%(input_list[i]))
            img = cv2.imread(opj(args.input_dir , input_list[i]))
            gt = cv2.imread(opj(args.gt_dir , gt_list[i]))
            img = image_align(img)
            gt = image_align(gt)
            result = predict_single(model,img)
            result = np.array(result, dtype = 'uint8')
            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)
            print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
            psnr_list.append(cur_psnr)
            ssim_list.append(cur_ssim)
        print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/num, cumulative_ssim/num))
        with open ('../try/psnr_list','wb') as fout:
            fout.write(pkl.dumps(psnr_list))
        with open ('../try/ssim_list','wb') as fout:
            fout.write(pkl.dumps(ssim_list))
        df = pd.DataFrame(np.array([psnr_list,ssim_list]).T, columns=['psnr','ssim'])
        df.head()
        print(df.apply(status))
    else:
        print ('Mode Invalid!')


def get_path(dir,home:str):
    path = opj(dir,home)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def write_interval(interval_dir,prefix,img,gt,result):
    raw_name = prefix+"_"+"input.png"
    gt_name = prefix+"_"+"gt.png"
    out_name = prefix+"_"+"output.png"
    cv2.imwrite( opj( get_path(interval_dir,'input'), raw_name), img )
    cv2.imwrite( opj( get_path(interval_dir,'gt'), gt_name), gt )
    cv2.imwrite( opj( get_path(interval_dir,'output'), out_name), result )

def split_result(args):
    # split the result with ssim (0,0.5);(0.5,0.82);(0.82,0.87);(0.87,1.0)
    split_point = [0.5, 0.82, 0.87]
    model = Generator().cuda()
    model.load_state_dict(torch.load(opj(args.model_dir, args.g_weights)))
    input_list = sorted(os.listdir(args.input_dir))
    gt_list = sorted(os.listdir(args.gt_dir))
    num = len(input_list)
    cumulative_psnr = 0
    cumulative_ssim = 0

    split_dir = args.split_dir
    interval1 = opj(split_dir,'interval_1')
    interval2 = opj(split_dir,'interval_2')
    interval3 = opj(split_dir,'interval_3')
    interval4 = opj(split_dir,'interval_4')
    interval_dirs = [interval1,interval2,interval3,interval4]
    for dir in interval_dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
    for i in range(num):
        print ('Processing image: %s'%(input_list[i]))
        img = cv2.imread(opj(args.input_dir , input_list[i]))
        gt = cv2.imread(opj(args.gt_dir , gt_list[i]))
        img = image_align(img)
        gt = image_align(gt)
        result = predict_single(model,img)
        result = np.array(result, dtype = 'uint8')
        cur_psnr = calc_psnr(result, gt)
        cur_ssim = calc_ssim(result, gt)
        print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim

        prefix = input_list[i].split('_')[0]
        if cur_ssim < split_point[0]:
            write_interval(interval_dirs[0],prefix,img,gt,result)
        elif cur_ssim < split_point[1]:
            write_interval(interval_dirs[1],prefix,img,gt,result)
        elif cur_ssim < split_point[2]:
            write_interval(interval_dirs[2],prefix,img,gt,result)
        else:
            write_interval(interval_dirs[3],prefix,img,gt,result)


    print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/num, cumulative_ssim/num))

def prepare(args):
    print("Not implement error.")

def run():
    args = get_args()
    print("Running with args:{}".format(args))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.prepare:
        prepare(args)
    elif args.train:
        print("Trianning")
        train(args)
    elif args.predict:
        predict(args)
    elif args.split:
        split_result(args)
    else:
        print("You won't run.")

if __name__ == "__main__":
    run()