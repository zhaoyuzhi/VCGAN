import os
import math
import random
import numpy as np
import cv2
from PIL import Image
from skimage import color
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size
        # Start from h1 and w1
        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)
        # Crop h1 ~ h2 and w1 ~ w2
        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        
    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1 : self.h2, self.w1 : self.w2, :]
        else:
            return img[self.h1 : self.h2, self.w1 : self.w2]

class ColorizationDataset(Dataset):
    def __init__(self, opt):
        # Note that:
        # 1. opt: all the options
        # 2. imglist: all the image names under "baseroot"
        self.opt = opt
        self.imglist = self.get_files(opt.baseroot)
    
    def get_files(self, path):
        # Read a folder, return the complete path
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))
        # Randomly sample the target slice
        sample_size = int(math.floor(len(ret) / self.opt.sample_size))
        ret = random.sample(ret, sample_size)
        # Re-arrange the list that meets multiplier of batchsize
        adaptive_len = int(math.floor(len(ret) / self.opt.batch_size) * self.opt.batch_size)
        ret = ret[:adaptive_len]
        return ret
    
    def img_aug(self, img):
        # Random scale
        '''
        if self.opt.geometry_aug:
            H_in = img.shape[0]
            W_in = img.shape[1]
            sc = np.random.uniform(self.opt.scale_min, self.opt.scale_max)
            H_out = int(math.floor(H_in * sc))
            W_out = int(math.floor(W_in * sc))
            # scaled size should be greater than opts.crop_size and remain the ratio of H to W
            if H_out < W_out:
                if H_out < self.opt.crop_size:
                    H_out = self.opt.crop_size
                    W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
            else: # W_out < H_out
                if W_out < self.opt.crop_size:
                    W_out = self.opt.crop_size
                    H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
            img = cv2.resize(img, (W_out, H_out))
        '''
        if self.opt.geometry_aug:
            H_in = img.shape[0]
            W_in = img.shape[1]
            if H_in < W_in:
                H_out = self.opt.crop_size
                W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
            else: # W_in < H_in
                W_out = self.opt.crop_size
                H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
            img = cv2.resize(img, (W_out, H_out))
        else:
            img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        # Random crop
        cropper = RandomCrop(img.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        img = cropper(img)
        '''
        # Random rotate
        if self.opt.angle_aug:
            # Rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                img = np.rot90(img, rotate)
            # Horizontal flip
            if np.random.random() >= 0.5:
                img = cv2.flip(img, flipCode = 1)
        '''
        return img

    def __getitem__(self, index):
        imgpath = self.imglist[index]                               # Path of one image
        # Read the images
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # RGB image
        grayimg = img[:, :, [0]] * 0.299 + img[:, :, [1]] * 0.587 + img[:, :, [2]] * 0.114
        grayimg = np.concatenate((grayimg, grayimg, grayimg), axis = 2)
        # Data augmentation
        grayimg = self.img_aug(grayimg)
        img = self.img_aug(img)
        # Normalized to [-1, 1]
        grayimg = np.ascontiguousarray(grayimg, dtype = np.float32)
        grayimg = (grayimg - 128) / 128
        img = np.ascontiguousarray(img, dtype = np.float32)
        img = (img - 128) / 128
        # To PyTorch Tensor
        grayimg = torch.from_numpy(grayimg).permute(2, 0, 1).contiguous()
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return grayimg, img
    
    def __len__(self):
        return len(self.imglist)

class MultiFramesDataset(Dataset):
    def __init__(self, opt, imglist, classlist):
        # If you want to use this code, please ensure:
        # 1. the structure of the training set should be given as: baseroot + classname (many classes) + imagename (many images)
        # 2. all the input images should be categorized, and each folder contains all the images of this class
        # 3. all the name of images should be given as: 00000.jpg, 00001.jpg, 00002.jpg, ... , 00NNN.jpg (if possible, not mandatary)
        # Input:
        # 1. opt: all the options
        # 2. imglist: all the "classname (many classes) + imagename (many images)"
        # 2. classlist: all the "classname (many classes)"
        # Note that:
        # 1. self.baseroot: the overall base
        # 2. self.classlist: the second base, it could not be used in get_item
        # 3. self.imgroot: the relative root for each image, and classified by categories (二维列表)
        # Inference:
        # number of classes = len(self.classlist) = len(self.imgroot)
        # number of images in a specific class = len(self.imgroot[k])
        # this dataset is fair for each class, not for each image, because the number of images in different classes is different
        self.opt = opt                                                  # baseroot is the base of all images
        self.imglist = imglist                                          # imglist should contain the category name of the series of frames + image names, in order
        self.classlist = classlist                                      # classlist should contain the category name of the series of frames
        self.imgroot = [list() for i in range(len(classlist))]          # imgroot contains the relative path of all images
        # Calculate the whole number of each class
        for i, classname in enumerate(self.classlist):
            for j, imgname in enumerate(imglist):
                if imgname.split('/')[-2] == classname:
                    self.imgroot[i].append(imgname)
        # Raise error
        for i in range(len(imglist)):
            if self.opt.iter_frames > len(imglist[i]):
                raise Exception("Your given iter_frames is too big for this training set!")

    def get_lab(self, imgpath):
        # Pre-processing, let all the images are in RGB color space
        img = cv2.imread(imgpath)                                       # read one image
        img = cv2.resize(img, (self.opt.crop_size_w, self.opt.crop_size_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                      # numpy RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C]
        # Convert RGB to Lab, finally get Tensor
        img = color.rgb2lab(img).astype(np.float32)                     # skimage Lab: L [0, 100], a [-128, 127], b [-128, 127], order [H, W, C]
        img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)).contiguous()      # Tensor Lab: L [0, 100], a [-128, 127], b [-128, 127], order [C, H, W]
        # Normaization
        l = img[[0], ...] / 50 - 1.0                                    # L, normalized to [-1, 1]
        ab = img[[1, 2], ...] / 110.0                                   # a and b, normalized to [-1, 1], approximately
        return l, ab

    def get_rgb(self, imgpath):
        # Pre-processing, let all the images are in RGB color space
        img = cv2.imread(imgpath)                                       # read one image
        img = cv2.resize(img, (self.opt.crop_size_w, self.opt.crop_size_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                      # numpy RGB: R [0, 255], G [0, 255], B [0, 255], order [H, W, C] 
        grayimg = img[:, :, [0]] * 0.299 + img[:, :, [1]] * 0.587 + img[:, :, [2]] * 0.114
        grayimg = np.concatenate((grayimg, grayimg, grayimg), axis = 2) # Grayscale image
        # Normalized to [-1, 1]
        grayimg = np.ascontiguousarray(grayimg, dtype = np.float32)
        grayimg = (grayimg - 128) / 128
        img = np.ascontiguousarray(img, dtype = np.float32)
        img = (img - 128) / 128
        # To PyTorch Tensor
        grayimg = torch.from_numpy(grayimg).permute(2, 0, 1).contiguous()
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return grayimg, img

    def __getitem__(self, index):
        # Choose a category of dataset, it is fair for each dataset to be chosen
        N = len(self.imgroot[index])
        # Pre-define the starting frame index in 0 ~ N - opt.iter_frames
        T = random.randint(0, N - self.opt.iter_frames)
        # Sample from T to T + opt.iter_frames
        in_part = []
        out_part = []
        for i in range(T, T + self.opt.iter_frames):
            imgpath = os.path.join(self.opt.baseroot, self.imgroot[index][i])
            l, ab = self.get_rgb(imgpath)
            in_part.append(l)
            out_part.append(ab)
        return in_part, out_part
    
    def __len__(self):
        return len(self.classlist)

if __name__ == "__main__":
    
    a = torch.randn(1, 3, 256, 256)
    b = a[:, [0], :, :] * 0.299 + a[:, [1], :, :] * 0.587 + a[:, [2], :, :] * 0.114
    b = torch.cat((b, b, b), 1)
    print(b.shape)

