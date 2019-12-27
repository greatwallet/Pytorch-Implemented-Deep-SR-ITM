# -*- coding: utf-8 -*-
# @Author  : Cheng Xiaotian
# @Email   : cxt16@mails.tsinghua.edu.cn

import numpy as np
import torch


def tensor_to_pil(tensor, mode):
    """ Convert a tensor (B, C, H, W) to ndarray (B, H, W, C)
    Args:
        pic (Tensor or numpy.ndarray) – Image to be converted to PIL Image.
        mode (string) - either be 'SDR' or 'HDR'
    """
    array = tensor.numpy()
    
    # check whether the tensor's shape is (C, H, W) or (B, C, H, W)
    if len(array.shape) == 3:
        array = np.moveaxis(array, 0, -1)
    elif len(array.shape) == 4:
        array = np.moveaxis(array, 1, -1)
    else:
        raise ValueError('Shape length should be {:d} or {:d} but received shape of {}'
                         .format(3, 4, array.shape))
    
    if mode == 'SDR':
        array *= 255.0
        array = array.astype(np.uint8)
        
    elif mode == 'HDR':
        array *= 1023.0
        array = array.astype(np.uint16)
        
    else:
        raise ValueError('Supposed to get {} or {}, but received {}'
                         .format('SDR', 'HDR', mode))
    return array
