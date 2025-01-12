## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

# from natsort import natsorted
from glob import glob
from basicsr.models.archs.nafcoanet_arch import NAFCoANet
from skimage import img_as_ubyte, io
from pdb import set_trace as stx


def warp(img,jit):
    jit = torch.from_numpy(-jit)
    # img = torch.from_numpy(img).float() # error: uint16 convert don't support
    img = torch.from_numpy(img.astype(np.int32)).float() # val psnr error, np.int32 pr np.float32 instead of np.int16
    h, w = img.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(img)
    grid.requires_grad = False

    grid_flow = grid + jit
    grid_flow = grid_flow.unsqueeze(0)
    grid_flow = grid_flow[:, :h, :w, :]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0  
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim = 3)

    img_tensor = unsqueeze_twice(img)
    # print(img_tensor,grid_flow)
    img_subdivision = F.grid_sample(img_tensor, grid_flow, 
        mode = 'bilinear', padding_mode = "reflection", align_corners = True) # nearest
    img = np.array(img_subdivision).astype(int)[0,0,:,:]
    return img

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='./realdata', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/workspace/workspace/chenzd/project/restore_sateline/experiments/Deblurring_NAFCoANet_20230511_480x640Dv1plus20uv_perceptual_nowarp/models//net_g_latest.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='RealData', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/test_Deblurring_NAFCoANet.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = NAFCoANet(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

# inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
# files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
inp_dir = os.path.join(args.input_dir)
print(inp_dir)
# files = natsorted(glob(os.path.join(inp_dir, '*.npy')))
files = glob(os.path.join(inp_dir, '*.jpg'))
print(len(files))
with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        
        # dic = np.load(file_, allow_pickle=True)
        # img_tdi = dic.item()['img_TDI']
        img_tdi = io.imread(file_, as_gray=True)
        print(img_tdi.shape)
        # img_tdi = np.expand_dims(img_tdi, axis=2)
        # print(img_tdi.shape)

        # img = np.float32(utils.load_img(file_))/255.
        # img = np.float32(img_tdi)/255.
        print(np.max(img_tdi), np.min(img_tdi))
        # print(img)
        img = torch.from_numpy(img_tdi).float().unsqueeze(0).unsqueeze(0)
        input_ = img.cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]
        # print(restored)

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
