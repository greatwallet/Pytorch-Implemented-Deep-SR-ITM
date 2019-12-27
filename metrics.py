# -*- coding: utf-8 -*-
# @Author  : Cheng Xiaotian
# @Email   : cxt16@mails.tsinghua.edu.cn

import torch

from torch.nn import Module
from torch.nn import functional as F

def psnr(input, target, peakval=None):
    r""" Creates a criterion that measures the Peak signal-to-noise ratio Error
    
    Please refer to wiki: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    # make sure the input shape must be as (B, C, H ,W)
    assert(len(input.shape) == 4)
    assert(target.size() == input.size())
     
    assert(input.dtype == target.dtype)
    # set peak value
    if peakval is None:
        if (input.dtype == torch.float16 or
           input.dtype == torch.float32 or
           input.dtype == torch.float64):
            peakval = 1.0
        elif input.dtype == np.uint8:
            peakval = 255
        elif input.dtype == np.int16:
            peakval = 65535
        else:
            raise ValueError("Unless the peakval is specified, \
                             the data type must be {}, {}, {}, \
                             {} and {}"
                             .format(torch.float16, torch.float32, 
                                     torch.float64, torch.uint8,
                                    torch.int16))
    
    squared_error = F.mse_loss(input, target, reduction='none')

    # add up the error and conduct mean on C, H, W
    mse = torch.mean(squared_error, (3, 2, 1))

    ret = 10 * torch.log10(peakval ** 2 / mse)
    ret = torch.mean(ret)
        
    return ret
    
class PSNR(Module):
    r""" Creates a criterion that measures the Peak signal-to-noise ratio Error
    
    Please refer to wiki: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    
    def __init__(self, peakval=None):
        super(PSNR, self).__init__()
        self.peakval = peakval
    
    def forward(self, input, target):
        return psnr(input, target, peakval=self.peakval)
        