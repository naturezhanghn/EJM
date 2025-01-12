import cv2
import torch
import numpy as np
import os.path as osp
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def calculate_lpips(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    
    if img1.shape[-1] == 1:
        mean = [0.5]
        std = [0.5]
    elif img1.shape[-1] == 3:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise ValueError(
            f'Expect the image channel 1 or 3, but got {img1.shape[-1]}.')
    img1, img2 = img2tensor([img1 / 255., img2 / 255.], bgr2rgb=True, float32=True)

    normalize(img1, mean, std, inplace=True)
    normalize(img2, mean, std, inplace=True)

    lpips_val = loss_fn_vgg(img1.unsqueeze(0).cuda(), img2.unsqueeze(0).cuda())
    return lpips_val