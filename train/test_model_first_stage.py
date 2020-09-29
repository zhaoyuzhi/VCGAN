# Test the model's first stage
import os
import argparse
import cv2
import numpy as np
from PIL import Image

def test(grayimg, model):
    # Forward and reshape to [H, W, C], in range [-1, 1]
    out_rgb = model(grayimg)
    out_rgb = out_rgb.squeeze(0).cpu().detach().numpy().reshape([3, 256, 256])
    out_rgb = out_rgb.transpose(1, 2, 0)
    # Return the original scale
    out_rgb = (out_rgb * 0.5 + 0.5) * 255
    out_rgb = out_rgb.astype(np.uint8)
    return out_rgb
    
def getImage(imgpath):
    # Read the images
    grayimg = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    grayimg = np.expand_dims(grayimg, 2)
    grayimg = np.concatenate((grayimg, grayimg, grayimg), axis = 2)
    # Normalized to [-1, 1]
    grayimg = (grayimg.astype(np.float64) - 128) / 128
    # To PyTorch Tensor
    grayimg = torch.from_numpy(grayimg.transpose(2, 0, 1).astype(np.float32)).contiguous()
    grayimg = grayimg.unsqueeze(0).cuda()
    return grayimg

def load_model(opt):
    model = network.FirstStageNet(opt)
    pretrained_dict = torch.load(opt.load_name)
    # Get the dict from processing network
    process_dict = model.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    model.load_state_dict(process_dict)
    model = model.cuda()
    return model

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # testing parameters
    parser.add_argument('--imgpath', type = str, default = 'C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256\\ILSVRC2012_val_00000016.JPEG', help = 'testing image path')
    parser.add_argument('--load_name', type = str, default = './First_Stage_epoch3_bs8.pth', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    parser.add_argument('--comparison', type = bool, default = True, help = 'compare with original RGB image or not')
    # GPU parameters
    parser.add_argument('--test_gpu', type = str, default = '0', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    # network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'in channel for U-Net encoder')
    parser.add_argument('--start_channels', type = int, default = 32, help = 'start channel for U-Net encoder')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'out channel for U-Net decoder')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'activation function for generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation function for discriminator')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    opt = parser.parse_args()
    
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.test_gpu
    print('Single-GPU mode, %s GPU is used' % (opt.test_gpu))
    
    # ----------------------------------------
    #                  Testing
    # ----------------------------------------
    import torch
    import network

    # Get image
    img = getImage(opt.imgpath)

    # Get model
    model = load_model(opt)

    # Get result [H, W, C], in range [0, 255]
    out_rgb = test(img, model)

    # Post-processing
    if opt.comparison:
        # Get result [H, W, C], in range [0, 255]
        ori_rgb = cv2.imread(opt.imgpath)[:, :, ::-1]
        out_rgb = np.concatenate((out_rgb, ori_rgb), axis = 1)
        
    # Print
    out_rgb = Image.fromarray(out_rgb)
    out_rgb.show()
    