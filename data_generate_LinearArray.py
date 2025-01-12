import numpy as np
import os
from numpy.random import random_sample
from numpy import sin
from random import random, gauss,seed
import torch.nn.functional as F
import cv2
import torch
import xlsxwriter as xw
# import h5py
import random
seed(0)

# Choose device 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)



def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg','.tif']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def list_image_files(directory):
    files = os.listdir(directory)
    img_list = [os.path.join(directory, f) for f in files if is_an_image_file(f)]
    name = [f for f in files]
    return img_list, name

def stretch_contrast(img_gray):
    img_gray_ = (img_gray - np.min(img_gray) ) / (np.max(img_gray) - np.min(img_gray))
    return img_gray_

def pre_process(img):
    # gamma correction
    img = np.array(img)
    # print(img.shape)
    if img.shape[2] == 3:
        img =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = stretch_contrast(img) 
    img = img / 255.0
    return img

def jitter(push_pixel, stage, T_per_stage_max, subdivision, f, amp, random_botton = True ):
    assert isinstance(subdivision, int) 
    sample_interval = T_per_stage_max / subdivision
    x = np.arange(0, (push_pixel + stage) * sample_interval * subdivision, sample_interval)
    if random_botton == True: 
        # f = [ random() * 2 * np.pi * fi for fi in f]
        f = [ gauss(mu=1, sigma=0.5) * 2 * np.pi * fi for fi in f]
        pha = random_sample(len(f)) * 2 * np.pi 
        
        amp = [ gauss(mu=1, sigma=0.05) * ampi for ampi in amp]  # TDI:random() / TDI_normal:random_factor 
        # amp = [ random() * ampi for ampi in amp]
        
        jix = 0
        for i in range(len(f)):
            jix += amp[i] * sin(f[i] * x + pha[i]) 
        # jix += np.random.normal(0, np.max(amp) * 0.05, size=jix.shape)
    else:
        f = [2 * np.pi * fi for fi in f]
        pha = [0 for i in f]
        amp = [ampi for ampi in amp]
        jix = 0
        for i in range(len(f)):
            jix  += amp[i] * sin(f[i] * x + pha[i]) 
    return torch.from_numpy(jix), torch.tensor([sample_interval])

def calc_para(H = 5*10**5, focal_length = 10, u_size = 5*10**(-6)):
    G = 6.674 * 10 ** (-11)
    M = 5.965 * 10 ** 24
    R = 6.371 * 10 ** 6
    v_sateline = np.sqrt( G * M / ( R + H ))
    v_groud = R / ( R + H ) * v_sateline
    v_scene = v_groud / H * focal_length
    # The velocity of the target image is equal to the velocity of charge transfer
    v_charge = v_scene 
    T_exposure_max = u_size / v_charge # Integral time
    return T_exposure_max

def unsqueeze_twice(x):
    return x.unsqueeze(0).unsqueeze(0)

def squeeze_twice(x):
    return x.squeeze(0).squeeze(0)

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref, data_range=255.0):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(data_range**2 / np.mean(np.square(diff)))
    return psnr

def gamma_correction(x,factor,inv=False):
    if inv:
        x = np.power(x, 1/factor)
    else:
        x = np.power(x, factor)
    return x 


def TDI_CCD(img, stage = 128, H = 5*10**5, focal_length = 10, u_size = 5*10**(-6),
        f_roll  = [ 1000, 100, 10, 1] , amp_roll  = [ 2, 2, 5, 5] ,
        f_pitch = [ 1000, 100, 10, 1] , amp_pitch = [ 1, 1, 2, 2] ,
        sigma_gauss = 0.17, lambda_poisson = 1.7e-3, subdivision = 6,
        jit_error = 0.3): # 
    """
    Jitter simulation.
    Noise simulation.
    """
    img = torch.from_numpy(img).float().cuda()
    h, w = img.shape
    
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(img)
    grid.requires_grad = False
    
    T_per_stage_max = calc_para(H, focal_length, u_size)

    push_pixel = w
    
    jit_roll, jit_roll_sample_interval  = jitter(push_pixel, stage, T_per_stage_max, subdivision, f=f_roll, amp=amp_roll,random_botton = True ) #.type_as(img) 
    jit_pitch, jit_pitch_sample_interval = jitter(push_pixel, stage, T_per_stage_max, subdivision, f=f_pitch, amp=amp_pitch,random_botton = True ) #.type_as(img)
    
    jit_accumulation_noise = torch.zeros_like(grid)
    
    # print(torch.max(jit_roll),torch.max(jit_pitch))
    
    jit_roll_noise = torch.normal(0, np.abs(torch.max(jit_roll)) * jit_error, size=jit_roll.shape) + jit_roll
    jit_pitch_noise = torch.normal(0, np.abs(torch.max(jit_pitch)) * jit_error, size=jit_roll.shape) + jit_pitch
    jit_roll_noise = jit_roll_noise.type_as(img) 
    jit_pitch_noise = jit_pitch_noise.type_as(img) 
    sumjit_x = 0
    sumjit_y = 0

    sumjit_x_clear = 0
    sumjit_y_clear = 0
    
    
    jit_roll, jit_pitch = jit_roll.type_as(img), jit_pitch.type_as(img) 
    assert jit_roll_sample_interval == jit_pitch_sample_interval    
    # jit_roll_sample_interval, jit_pitch_sample_interval = jit_pitch_sample_interval.type_as(img), jit_pitch_sample_interval.type_as(img)  
    
    img_accumulation = unsqueeze_twice(torch.zeros_like(img))
    jit_accumulation = torch.zeros_like(grid)

    for i in range(stage):
        img_one_stage = torch.zeros_like(img_accumulation)
        for n in range(subdivision):
            ########### jit real ##################
            jit_x = jit_roll[i * subdivision + n : (i + w ) * subdivision + n : subdivision]
            jit_y = jit_pitch[i * subdivision + n : (i + w ) * subdivision + n  : subdivision]

            sumjit_x_clear += jit_x
            sumjit_y_clear += jit_y

            jit_x = jit_x.repeat(h, 1)
            jit_y = jit_y.repeat(h, 1)

            grid_jiy, grid_jix = jit_x,jit_y
            jit = torch.stack((grid_jix, grid_jiy), 2) 
            jit_accumulation += jit 

            ########### jit noise ##################
            jit_x_noise = jit_roll_noise[i * subdivision + n : (i + w ) * subdivision + n : subdivision]
            jit_y_noise = jit_pitch_noise[i * subdivision + n : (i + w ) * subdivision + n  : subdivision]
            
            sumjit_x += jit_x_noise
            sumjit_y += jit_y_noise

            jit_x_noise = jit_x_noise.repeat(h, 1)
            jit_y_noise = jit_y_noise.repeat(h, 1)

            grid_jiy_noise, grid_jix_noise = jit_x_noise,jit_y_noise
            jit_noise = torch.stack((grid_jix_noise, grid_jiy_noise), 2) 
            jit_accumulation_noise += jit_noise 


            img_tensor = unsqueeze_twice(img)
            
            grid_flow = grid + jit
            grid_flow = grid_flow.unsqueeze(0)
            grid_flow = grid_flow[:, :h, :w, :]
            grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0  
            grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
            grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim = 3)
            
            img_subdivision = F.grid_sample(img_tensor, grid_flow, 
                mode = 'bilinear', padding_mode = "reflection", align_corners = True) # nearest
            img_one_stage += img_subdivision / subdivision 
        
        noise = torch.normal(0, sigma_gauss, [h, w]).cuda() \
              + torch.poisson(lambda_poisson * img_one_stage) 
        img_accumulation += img_one_stage + noise
        
    img_out = img_accumulation / stage
    jit_information = jit_accumulation / (stage * subdivision)
    img_out= squeeze_twice(img_out)
    
    img_out = np.array(img_out.cpu()) 
    jit_information = np.array( jit_information.cpu() )
    jit_roll_information = np.array(jit_roll.cpu())
    jit_pitch_information = np.array(jit_pitch.cpu())
    
    
    jit_information_noise = jit_accumulation_noise / (stage * subdivision)
    jit_information_noise = np.array( jit_information_noise.cpu() )
    sumjit_y /= (stage * subdivision)
    sumjit_x /= (stage * subdivision)
    eq_jit_roll = np.array(sumjit_x.cpu())
    eq_jit_pitch = np.array(sumjit_y.cpu())

    jit_roll_noise = np.array(jit_roll_noise.cpu())
    jit_pitch_noise = np.array(jit_pitch_noise.cpu())

    sumjit_x_clear /= (stage * subdivision)
    sumjit_y_clear /= (stage * subdivision)
    sumjit_x_clear = np.array(sumjit_x_clear.cpu())
    sumjit_y_clear = np.array(sumjit_y_clear.cpu())
    
    return img_out, jit_information, jit_roll_information, jit_pitch_information, jit_roll_sample_interval, jit_information_noise ,\
            eq_jit_roll, eq_jit_pitch,jit_roll_noise, jit_pitch_noise,sumjit_x_clear ,sumjit_y_clear 

def record_excel( excel=None, data=None, excel_row_num=None, mode=None ,init=None):
    if init is not None:
        path = init + mode  + "eval.xlsx"
        workbook  = xw.Workbook(path )
        worksheet = workbook.add_worksheet("evaluation result")
        worksheet.activate()  # 激活表
        title = ['name', 'psnr', 'ssim'] 
        worksheet.write_row('A1', title) 
        excel = [workbook,worksheet]
        return excel
    elif mode == "close":
        excel[0].close()
    else:
        try:
            excel_row_name = "A" + str(excel_row_num )
            excel[1].write_row(excel_row_name, data)
        except:
            pass 

if __name__ == "__main__":
    dataset_name = "TDI_dataset_td_LinearArray"
    p = ["test","train","val"]
    div_data_flag = 0
    val_id = 1
    train_id = 1
    test_id = 1

    # make dir
    excel_dict = {}
    for pi in p:
        mk_save_dir = "./" + dataset_name + "/tdi_images/" + pi + "/"
        mk_eval_dir = "./" + dataset_name + "/eval/" + pi + "/"
        mk_dataset_dir = "./" + dataset_name + "/tdi_dataset/" + pi + "/"
        os.makedirs(mk_save_dir, exist_ok=True) 
        os.makedirs(mk_eval_dir, exist_ok=True)  
        os.makedirs(mk_dataset_dir, exist_ok=True) 
        excel = record_excel( data=None, excel_row_num=None, mode=pi, init = mk_eval_dir  )
        excel_dict[pi] = excel

    
    path1 = "/workspace/workspace/chenzd/zipfile/AerialImageDataset/test/images/"
    path2 = "/workspace/workspace/chenzd/zipfile/AerialImageDataset/train/images/"
    # path3= "/data/zhangzr/dotav1/dotav1-all/"
    
    path = [path1, path2]
    for k,datapath in enumerate(path):
        print("This",k,datapath)
        img_path = datapath
        img_h,img_w = 480, 640
        
        image_list,name = list_image_files(img_path)
        img_num = len(image_list)
        image_list = sorted(image_list)
        
        excel_row_num = 2
        for n in range(0, img_num): 
            num = n + 1
            print(f"Image {k}_{num} is being processed.")
            try:
                img_gray = cv2.imread(image_list[n])
                img = pre_process(img_gray)
            except:
                print(f"Image {k}_{num} failed to process.")
                continue
            if np.sum(img_gray ) < 3*(img_h*img_w):
                continue

            height, width = img.shape 
            for i in range(0, height, img_h):
                for j in range(0, width, img_w):
                    if i + img_h > height or j + img_w > width: 
                        continue
                    img_crop = img[i : i + img_h, j : j + img_w]
                    if (img_h,img_w) != img_crop.shape:
                        print("crop error:",img_crop.shape)
                        continue
                    img_crop_gamma = gamma_correction(img_crop.copy() , 1/2.2 ,inv=True)
                    img_TDI, jit_information, jit_roll_information, jit_pitch_information, sample_interval, jit_information_noise ,\
                       eq_jit_roll, eq_jit_pitch,jit_roll_noise, jit_pitch_noise, eq_jit_roll_clear, eq_jit_pitch_clear \
                        = TDI_CCD(img_crop_gamma , stage = 1, H = 5*10**5, focal_length = 10, u_size = 5*10**(-6),
                                    f_roll  = [ 1000, 2000, 3000, 4000] , amp_roll  = [ 4, 1.5, 1.0, 0.5] ,
                                    f_pitch = [ 1000, 2000, 3000, 4000] , amp_pitch = [ 1, 0.5, 0.3, 0.2] ,
                                    sigma_gauss = 0.01, lambda_poisson = 0.01 * 1e-2, subdivision = 6, jit_error = 0.2)
                    
                    # flow = draw_hsv(jit_information*10)
                    flow = flow_to_image(jit_information)
                    flow_noise = flow_to_image(jit_information_noise)
                    
                    # img_TDI = gamma_correction(img_TDI , 1/2.2 ,inv=False)
                    
                    img_crop = (np.clip(img_crop,0,1)*255*255).astype(np.uint16)
                    img_TDI = (np.clip(img_TDI,0,1)*255*255).astype(np.uint16)
                    # print(np.min(img_crop),np.max(img_crop),np.min(img_TDI),np.max(img_TDI))

                    tdi_psnr = calculate_psnr(img_TDI, img_crop )
                    tdi_ssim = calculate_ssim(img_TDI, img_crop)

                    TDI_data_dict = {
                        "img_TDI" : img_TDI,
                        "img_gt" : img_crop,
                        "jit_information" : jit_information,
                        "jit_roll_information" : jit_roll_information,
                        "jit_pitch_information" : jit_pitch_information,
                        "tdi_psnr" : tdi_psnr,
                        "tdi_ssim" : tdi_ssim ,
                        "jit_information_noise" : jit_information_noise ,
                        "eq_jit_roll" : eq_jit_roll, 
                        "eq_jit_pitch" : eq_jit_pitch,
                        "jit_roll_noise":jit_roll_noise,
                        "jit_pitch_noise":jit_pitch_noise,
                        "eq_jit_roll_clear":eq_jit_roll_clear ,
                        "eq_jit_pitch_clear" :eq_jit_pitch_clear} 
                    
                    div_data_flag += 1 
                    save_name = f"{k}_{num}_{i}_{j}"
                    out_put = [save_name, tdi_psnr,tdi_ssim]  
                    if div_data_flag % 10 == 1:
                        data_div = "val"
                        val_id += 1
                        record_excel( excel=excel_dict[data_div], data=out_put , excel_row_num=val_id, mode=data_div )
                    elif div_data_flag % 10 == 2:
                        data_div = "test"
                        test_id += 1
                        record_excel( excel=excel_dict[data_div], data=out_put , excel_row_num=test_id, mode=data_div )
                    else:
                        data_div = "train"
                        train_id += 1 
                        record_excel( excel=excel_dict[data_div], data=out_put , excel_row_num=train_id, mode=data_div )
                    
                    img_TDI = gamma_correction(img_TDI , 1/2.2 ,inv=False)
                        
                    save_dir = "./" + dataset_name + "/tdi_images/" + data_div + "/"
                    eval_dir = "./" + dataset_name + "/eval/" + data_div  + "/"
                    dataset_dir = "./" + dataset_name + "/tdi_dataset/" + data_div  + "/"

                    np.save(dataset_dir + f"/{save_name}_TDI_data.npy", TDI_data_dict )
                    cv2.imwrite(save_dir + f"/{save_name}_flow.png" , flow )
                    cv2.imwrite(save_dir + f"/{save_name}_flow_noise.png" , flow_noise )
                    cv2.imwrite(save_dir + f"/{save_name}_sharp.png", img_crop )
                    cv2.imwrite(save_dir + f"/{save_name}_blur.png" , img_TDI )
                    
                    # hf = h5py.File(dataset_dir + f"/{num}_{i}_{j}_TDI_data.h5", 'w')
                    # for key in TDI_data_dict.keys():
                    #     hf.create_dataset( key , data = TDI_data_dict[key])
                    # hf.close()  
            # break              
    for pi in p:
        record_excel( excel=excel_dict[pi], mode="close" )


        

    
