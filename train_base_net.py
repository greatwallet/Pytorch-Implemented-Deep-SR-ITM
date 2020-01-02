# -*- coding: utf-8 -*-
# @Author  : Cheng Xiaotian
# @Email   : cxt16@mails.tsinghua.edu.cn

import json
import numpy as np
import os
import os.path as osp
import time
import torch
import torchvision

from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import YouTubeDataset
from metrics import MS_SSIM, PSNR, SSIM
from models import SR_ITM_base_net

# Global Parameters
# root directory to data
root_dir = 'data'

# checkpoint directory
checkpoint_dir = osp.join('checkpoints', 'base_net')
if not osp.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
# batch size
batch_size = 16

# number of workers
num_workers = 2

# feature channels in the model
feature_channels = 64

# Upsample scale
scale = 2

# learning rate
lr = 5e-7

# weight decay
weight_decay = 5e-4

# start epoch
start_epoch = 1

# max epochs
max_epochs = 200

# whether to use cuda
use_cuda = True

# gpu id
gpu_id = 3

# if load from the checkpoint 
load = False

# if resume from the checkpoint
resume = False

# check epoch if resume is True
check_epoch = 100

# display interval
disp_interval = 100

# indent of json file
indent = 4

# whether validate
validate = True

# seeds
torch_seed = 2020
numpy_seed = 2020

def train_base_net(net, trainloader, testloader, optimizer, criterions, device):
    mse = criterions['mse']
    psnr = criterions['psnr']
    ssim = criterions['ssim']
    ms_ssim = criterions['ms_ssim']
    
    train_metric = {
        'mse': [], 
        'psnr': []
    }
    
    test_metric = {
        'mse': [], 
        'psnr': [], 
        'ssim': [], 
        'ms_ssim': []
    } if validate else None
    
    for epoch in range(start_epoch, max_epochs + 1):  # loop over the dataset multiple times
        # training phase
        net.train()
        running_loss = 0.0
        running_psnr_val = 0.0
        start = time.time()
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            SDR_img, HDR_img, SDR_base = data
            SDR_img = SDR_img.to(device)
            HDR_img = HDR_img.to(device)
            SDR_base = SDR_base.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(SDR_img, SDR_base)
            loss = mse(outputs, HDR_img)
            psnr_val = psnr(outputs, HDR_img)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_psnr_val += psnr_val.item()
            
            if i % disp_interval == disp_interval - 1: 
                end = time.time()
                mean_train_loss = running_loss / disp_interval
                mean_train_psnr_val = running_psnr_val / disp_interval
                
                train_metric['mse'].append(mean_train_loss)
                train_metric['psnr'].append(mean_train_psnr_val)
                
                print('[epoch %3d / %3d, iter %5d / %5d] MSE: %.4e, PSNR: %.4f, lr: %.2e, time cost: %f' %
                      (epoch, max_epochs, i + 1, len(trainloader), mean_train_loss, 
                       mean_train_psnr_val, lr, end - start))
                running_loss = 0.0
                running_psnr_val = 0.0
                start = time.time()
                
        # validation phase
        if validate is True:
            if use_cuda is True and torch.cuda.is_available():
                torch.cuda.empty_cache()

            net.eval()
            running_loss = 0.0
            running_psnr_val = 0.0
            running_ssim_val = 0.0 
            running_ms_ssim_val = 0.0
            
            start = time.time()
            print('>>> Testing...')

            with torch.no_grad():
                for i, data in enumerate(testloader, 0):
                    # get the inputs
                    SDR_img, HDR_img, SDR_base = data
                    SDR_img = SDR_img.to(device)
                    HDR_img = HDR_img.to(device)
                    SDR_base = SDR_base.to(device)

                    # forward
                    outputs = net(SDR_img, SDR_base)
                    loss = mse(outputs, HDR_img)
                    psnr_val = psnr(outputs, HDR_img)
                    ssim_val = ssim(outputs, HDR_img)
                    ms_ssim_val = ms_ssim(outputs, HDR_img)

                    # print statistics
                    running_loss += loss.item()
                    running_psnr_val += psnr_val.item()
                    running_ssim_val += ssim_val.item()
                    running_ms_ssim_val += ms_ssim_val.item()

                end = time.time()
                mean_test_loss = running_loss / len(testloader)
                mean_test_psnr_val = running_psnr_val / len(testloader)
                mean_test_ssim_val = running_ssim_val / len(testloader)
                mean_test_ms_ssim_val = running_ms_ssim_val / len(testloader)

                test_metric['mse'].append(mean_test_loss)
                test_metric['psnr'].append(mean_test_psnr_val)
                test_metric['ssim'].append(mean_test_ssim_val)
                test_metric['ms_ssim'].append(mean_test_ms_ssim_val)
                
                print('[epoch %d] MSE: %.4e, PSNR: %.4f, SSIM: %.4f, MS_SSIM: %.4f, lr: %.2e, time cost: %f' %
                          (epoch, mean_test_loss, 
                           mean_test_psnr_val, 
                           mean_test_ssim_val, 
                           mean_test_ms_ssim_val, 
                           lr, end - start))

                running_loss = 0.0
                running_psnr_val = 0.0
                start = time.time()

        save_name = osp.join(checkpoint_dir, 'base_net_{:03d}.pth'
                                 .format(epoch))
        torch.save({
            'epoch': epoch, 
            'model': net.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'train_loss': mean_train_loss, 
            'train_psnr': mean_train_psnr_val, 
            'test_loss': None if validate is False else mean_test_loss, 
            'test_psnr': None if validate is False else mean_test_psnr_val, 
            'test_ssim': None if validate is False else mean_test_ssim_val, 
            'test_ms_ssim': None if validate is False else mean_test_ms_ssim_val, 
        }, save_name)
        print('save model: {}'.format(save_name))
        
    return train_metric, test_metric
           
if __name__ == '__main__':
    # set seed
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)
    
    if torch.cuda.is_available() and not use_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with `use_cuda = True`")
    
    # load data
    trainset = YouTubeDataset(osp.join(root_dir, 'trainset_SDR'),
                              osp.join(root_dir, 'trainset_HDR'), 
                              phase='train')
    
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    
    testset = YouTubeDataset(osp.join(root_dir, 'testset_SDR'),
                              osp.join(root_dir, 'testset_HDR'), 
                              phase='test', scale=scale)
    
    testloader = DataLoader(testset, batch_size=1, 
                            shuffle=False, num_workers=num_workers)
    # load base net
    base_net= SR_ITM_base_net(channels=feature_channels, scale=scale)
    
    criterions = {
        'mse': nn.MSELoss(reduction='mean'), 
        'psnr': PSNR(peakval=1.0), 
        'ssim': SSIM(data_range=1.0), 
        'ms_ssim': MS_SSIM(data_range=1.0)
    }
    
    optimizer = optim.Adam(base_net.parameters(), lr=lr, 
                           weight_decay=weight_decay)
    
    device = torch.device("cuda:%d" % gpu_id if use_cuda else "cpu")
    torch.cuda.set_device(device)
    
    if use_cuda is True:
        base_net.cuda()
         
    if load is True or resume is True:
        load_name = osp.join(checkpoint_dir, 
                            'base_net_{}.pth'.format(check_epoch))
        print('loading checkpoint: {}'.format(load_name))
        checkpoint = torch.load(load_name)
        base_net.load_state_dict(checkpoint['model'])
        
        if load is False and resume is True:
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            weight_decay = optimizer.param_groups[0]['weight_decay']
            
    train_metric, test_metric = train_base_net(net=base_net, 
                                               trainloader = trainloader, 
                                               testloader=testloader, 
                                               optimizer=optimizer, 
                                               criterions=criterions, 
                                               device=device)
    # save result
    now = datetime.now() # current date and time
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    time = now.strftime("%H_%M_%S")
    
    res_name = osp.join(checkpoint_dir, 'base_net_{}_{}_{}_{}.json'.format(year, month, day, time))
    hyperparameters = {
        'batch_size': batch_size, 
        'num_workers': num_workers, 
        'feature_channels': feature_channels, 
        'scale': scale, 
        'lr': lr, 
        'weight_decay': weight_decay, 
        'start_epoch': start_epoch, 
        'max_epochs': max_epochs
    }
    metrics = {
        'train_metric': train_metric, 
        'test_metric': test_metric
    }
    res = {
        'file_path': res_name, 
        'date': '{}_{}_{}_{}'.format(year, month, day, time), 
        'data_dir': root_dir,
        'checkpoint_dir': checkpoint_dir, 
        'use_cuda': use_cuda, 
        'device': str(device), 
        'load_checkpoint': load, 
        'resume_checkpoint': resume, 
        'check_epoch': check_epoch, 
        'display_interval': disp_interval, 
        'validate': validate, 
        'numpy_seed': numpy_seed, 
        'torch_seed': torch_seed, 
        'hyperparameters': hyperparameters, 
        'metrics': metrics
    }
    with open(res_name, 'w') as f:
        json.dump(res, f, indent=indent)
