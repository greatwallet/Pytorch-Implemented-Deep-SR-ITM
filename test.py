# -*- coding: utf-8 -*-
# @Author  : Cheng Xiaotian
# @Email   : cxt16@mails.tsinghua.edu.cn

import cv2
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
from models import SR_ITM_full_net
from utils import tensor_to_pil

# Global Parameters
# testset path
testset_SDR_dir = "./data/testset_SDR"

# batch size of dataloader
batch_size = 1

# number of workers
num_workers = 2

# model path
model_path = "./checkpoints/full_net/full_net_270.pth" 

# output path: if `None` then results will not be saved
output_path = "./results"
if not osp.exists(output_path):
    os.makedirs(output_path)

# file type of results
file_type = "png"
    
# feature channels in the model
feature_channels = 64

# Upsample scale
scale = 2

# whether to use cuda
use_cuda = True

# gpu id
gpu_id = 1

# validate or not
validate = True

# if `validate` == True, please specify `testset_HDR_dir`
testset_HDR_dir = "./data/testset_HDR"

# seeds
torch_seed = 2020
numpy_seed = 2020

# verbose 
verbose = True

def test(net, testloader, criterions, device):
    if validate:
        mse = criterions['mse']
        psnr = criterions['psnr']
        ssim = criterions['ssim']
        ms_ssim = criterions['ms_ssim']
    
    if use_cuda is True and torch.cuda.is_available():
        torch.cuda.empty_cache()

    net.eval()
    
    running_mse_val = 0.0
    running_psnr_val = 0.0
    running_ssim_val = 0.0 
    running_ms_ssim_val = 0.0

    start = time.time()
    idx = 1
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            SDR_img, HDR_img, SDR_base = data
            SDR_img = SDR_img.to(device)
            SDR_base = SDR_base.to(device)

            # forward
            outputs = net(SDR_img, SDR_base)
            
            if validate:
                HDR_img = HDR_img.to(device)
                mse_val = mse(outputs, HDR_img)
                psnr_val = psnr(outputs, HDR_img)
                ssim_val = ssim(outputs, HDR_img)
                ms_ssim_val = ms_ssim(outputs, HDR_img)  
                
                running_mse_val += mse_val.item()
                running_psnr_val += psnr_val.item()
                running_ssim_val += ssim_val.item()
                running_ms_ssim_val += ms_ssim_val.item()

            if output_path is not None:
                outputs_numpy = tensor_to_pil(outputs.cpu(), mode="HDR")
                for j in range(outputs_numpy.shape[0]):
                    img_out_array = outputs_numpy[j, :, :, :]
                    img_name = osp.join(output_path, '{:06d}.{}'.format(idx, file_type))
                    cv2.imwrite(img_name, img_out_array)
                    idx += 1
                    
                    if verbose:
                        print("Save image to {}".format(img_name))
                    
            
        end = time.time()
        if validate:
            mean_test_mse_val = running_mse_val / len(testloader)
            mean_test_psnr_val = running_psnr_val / len(testloader)
            mean_test_ssim_val = running_ssim_val / len(testloader)
            mean_test_ms_ssim_val = running_ms_ssim_val / len(testloader)
            
            print('MSE: %.4e, PSNR: %.4f, SSIM: %.4f, MS_SSIM: %.4f, time cost: %f' %
              (mean_test_mse_val, 
               mean_test_psnr_val, 
               mean_test_ssim_val, 
               mean_test_ms_ssim_val, 
               end - start))
               
if __name__ == '__main__':
    # set seed
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)
        
    device = torch.device("cuda:%d" % gpu_id if use_cuda else "cpu")
    torch.cuda.set_device(device)
    
    if torch.cuda.is_available() and not use_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with `use_cuda = True`")
    
    # load data
    testset = YouTubeDataset(
        SDR_dir=testset_SDR_dir,
        HDR_dir=None if not validate else testset_HDR_dir,
        phase="test" if not validate else "val", 
        scale = None if not validate else scale
    )
    data_amount = len(testset)
    testloader = DataLoader(testset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    # load full net
    full_net= SR_ITM_full_net(channels=feature_channels, scale=scale)
    print('loading checkpoint: {}'.format(model_path))
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    full_net.load_state_dict(checkpoint['model']) 
    
    if use_cuda is True:
        full_net.cuda()
    
    criterions = {
        'mse': nn.MSELoss(reduction='mean'), 
        'psnr': PSNR(peakval=1.0), 
        'ssim': SSIM(data_range=1.0), 
        'ms_ssim': MS_SSIM(data_range=1.0)
    }
            
    test(
        net=full_net, 
        testloader=testloader, 
        criterions=criterions, 
        device=device
    )
    