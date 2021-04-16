# Note that this warp error is implemented using Python 3.6 and PyTorch 1.0
# Please build an appropriate environment for the PWC-Net to compute optical flow
import argparse
import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pwcnet

# Compute the optical flow
def compute_flow_magnitude(flow):
    flow_mag = flow[0, :, :] ** 2 + flow[1, :, :] ** 2
    return flow_mag

def compute_flow_gradients(flow):
    # initial u, v for horizontal and vertical gradients
    H = flow.shape[1]
    W = flow.shape[2]
    flow_x_du = torch.zeros((H, W))
    flow_x_dv = torch.zeros((H, W))
    flow_y_du = torch.zeros((H, W))
    flow_y_dv = torch.zeros((H, W))
    # compute gradients
    flow_x = flow[0, :, :]
    flow_y = flow[1, :, :]
    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]
    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv

# For the parameters, please reference this paper:
# Sundaram, N., Brox, T., & Keutzer, K. (2010, September).
# Dense point trajectories by GPU-accelerated large displacement optical flow.
# In European conference on computer vision (pp. 438-451). Springer, Berlin, Heidelberg.
# 1 = occlusion / without disocclusion; 0 = non-occlusion
def define_occlusion(flownet, tensor1, tensor2):
    # forward flow in [-20, 20], dimension [2, H, W]
    forward_flow = pwcnet.PWCEstimate(flownet, tensor1, tensor2, drange = True, reshape = True)
    forward_flow = forward_flow.squeeze(0)
    # backward flow in [-20, 20], dimension [2, H, W]
    backward_flow = pwcnet.PWCEstimate(flownet, tensor2, tensor1, drange = True, reshape = True)
    backward_flow = backward_flow.squeeze(0)
    ### Occlusion
    # Equation (29) left
    forward_backward_flow = forward_flow + backward_flow
    equ29_left = compute_flow_magnitude(forward_backward_flow)
    # Equation (29) right
    forward_flow_mag = compute_flow_magnitude(forward_flow)
    backward_flow_mag = compute_flow_magnitude(backward_flow)
    equ29_right = 0.01 * (forward_flow_mag + backward_flow_mag) + 0.5
    # Equation (29) result
    equ29_mask = equ29_left > equ29_right
    ### Motion Boundary
    # Equation (30) left
    flow_x_du, flow_x_dv, flow_y_du, flow_y_dv = compute_flow_gradients(backward_flow)
    u_mag = flow_x_du ** 2 + flow_x_dv ** 2
    v_mag = flow_y_du ** 2 + flow_y_dv ** 2
    equ30_left = u_mag + v_mag
    # Equation (30) right
    equ30_right = 0.01 * backward_flow_mag + 0.002
    equ30_left = equ30_left.cuda()
    equ30_mask = equ30_left > equ30_right
    ### Combine masks
    equ29_mask = equ29_mask.type(torch.uint8)
    equ30_mask = equ30_mask.type(torch.uint8)
    occlusion = equ29_mask | equ30_mask
    occlusion = occlusion.type(torch.FloatTensor)
    count = torch.sum(occlusion)
    return occlusion, count, forward_flow

# Return the whole paths
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

# read a txt expect EOF
def text_readlines(filename):
    # try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def define_imglist(opt):
    # wholepathlist contains: basepath + class_name + image_name, while the input is basepath
    wholepathlist = text_readlines(opt.imagelist_txt)
    # classlist contains all class_names
    classlist = text_readlines(opt.class_txt)
    print('There are %d classes in the testing set:' % len(classlist), classlist)
    # imglist contains all class_names + image_names
    # imglist first dimension: class_names
    # imglist second dimension: basepath + class_name + image_name, for the curent class
    imglist = [list() for i in range(len(classlist))]
    for i, classname in enumerate(classlist):
        for j, imgname in enumerate(wholepathlist):
            if imgname.split('/')[-2] == classname:
                imgname = os.path.join(opt.basepath, 'DAVIS', imgname)
                imglist[i].append(imgname)
    return imglist

# Read an image and return a tensor [1, 3, H, W]
def read_img(path):
    img = cv2.imread(path)
    img = img[:, :, ::-1] / 255.0
    img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)).contiguous()
    img = img.unsqueeze(0).cuda()
    return img

# Create PWC-Net
def create_pwcnet(opt):
    # Initialize the network
    flownet = pwcnet.PWCNet().eval()
    # Load a pre-trained network
    data = torch.load(opt.pwcnet_path)
    if 'state_dict' in data.keys():
        flownet.load_state_dict(data['state_dict'])
    else:
        flownet.load_state_dict(data)
    print('PWCNet is loaded!')
    # It does not gradient
    for param in flownet.parameters():
        param.requires_grad = False
    return flownet

def save_report(content, filename, mode = 'a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    file.write(str(content) + '\n')
    file.close()

if __name__ == "__main__":

    # Define the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', type = str, \
        default = '', \
            help = 'the path contains all the generated frames')
    parser.add_argument('--imagelist_txt', type = str, default = './DAVIS_test_imagelist_without_first_frame.txt', help = 'the path contains all the relative image names')
    parser.add_argument('--class_txt', type = str, default = './DAVIS_test_class.txt', help = 'the path contains all the class names')
    parser.add_argument('--pwcnet_path', type = str, default = './pwcNet-default.pytorch', help = 'the path contains pre-trained PWC-Net')
    parser.add_argument('--report', type = bool, default = False, help = 'whether the results should be saved')
    parser.add_argument('--report_txt', type = str, default = './report.txt', help = 'the path saving report')
    opt = parser.parse_args()
    print(opt)

    # Create PWC-Net
    flownet = create_pwcnet(opt)
    flownet = flownet.cuda()

    # Get the 2-dimensional list, containing whole path of each image
    loss = 0
    loss_count = 0
    imglist = define_imglist(opt)

    #print('Calculating...')
    # Loop all the paths
    for i in range(len(imglist)):
        # compute the number of images in current class
        img_num = len(imglist[i])
        for j in range(img_num - 1):
            first_path = imglist[i][j]
            second_path = imglist[i][j + 1]
            first_tensor = read_img(first_path)                                         # first_tensor: dimension [1, 3, H, W]; range [0, 1]
            second_tensor = read_img(second_path)                                       # second_tensor: dimension [1, 3, H, W]; range [0, 1]
            # occlusion: dimension [H, W]; range 0 or 1     count: number of 1 in occlusion     forward_flow: dimension [2, H, W]; range [-20, 20]
            occlusion, count, forward_flow = define_occlusion(flownet, first_tensor, second_tensor)
            occlusion = occlusion.unsqueeze(0).unsqueeze(0)
            occlusion = torch.cat((occlusion, occlusion, occlusion), dim = 1)           # occlusion: dimension [1, 3, H, W]; range 0 or 1
            forward_flow = forward_flow.unsqueeze(0)                                    # forward_flow: dimension [1, 2, H, W]; range [-20, 20]
            # warp the first image
            second_warpped = pwcnet.PWCNetBackward(first_tensor, forward_flow)          # second_warpped: dimension [1, 3, H, W]; range [0, 1]
            # count loss
            non_occlusion = 1 - occlusion
            non_occlusion = non_occlusion.cuda()
            HW = non_occlusion.shape[2] * non_occlusion.shape[3]
            scaled_mse_loss = F.mse_loss(second_tensor * non_occlusion, second_warpped * non_occlusion, reduction = 'sum') / (HW - count)
            print('Now it is the %d-th category and %d-th pair: [occlusion points %d] [loss %.8f]' % (i, j, count.data, scaled_mse_loss.item()))
            # counting
            loss = loss + scaled_mse_loss
            loss_count = loss_count + 1

    # Compute average loss
    avg_loss = loss / loss_count
    print(opt.basepath)
    print('The average loss: %.8f' % (avg_loss.item()))

    # Report to a new txt file
    if opt.report:
        content = '%s \t wrap error: %8f' % (os.path.split(opt.basepath)[-2], avg_loss.item())
        save_report(content, opt.report_path)
    