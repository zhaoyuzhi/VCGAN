import argparse
import numpy as np
import os
import cv2
from skimage import io
from skimage import measure
from skimage import transform
from skimage import color

# Compute the mean-squared error between two images
def MSE(srcpath, dstpath, gray2rgb = False, scale = True):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.concatenate((dst, dst, dst), axis = 2)
    if scale:
        resize_h = scr.shape[0]
        resize_w = scr.shape[1]
        dst = cv2.resize(dst, (resize_w, resize_h))
    else:
        resize_h = dst.shape[0]
        resize_w = dst.shape[1]
        scr = cv2.resize(scr, (resize_w, resize_h))
    mse = measure.compare_mse(scr, dst)
    return mse

# Compute the normalized root mean-squared error (NRMSE) between two images
def NRMSE(srcpath, dstpath, gray2rgb = False, scale = True, mse_type = 'Euclidean'):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.expand_dims(dst, axis = 2)
        dst = np.concatenate((dst, dst, dst), axis = 2)
    if scale:
        resize_h = scr.shape[0]
        resize_w = scr.shape[1]
        dst = cv2.resize(dst, (resize_w, resize_h))
    else:
        resize_h = dst.shape[0]
        resize_w = dst.shape[1]
        scr = cv2.resize(scr, (resize_w, resize_h))
    nrmse = measure.compare_nrmse(scr, dst, norm_type = mse_type)
    return nrmse

# Compute the peak signal to noise ratio (PSNR) for an image
def PSNR(srcpath, dstpath, gray2rgb = False, scale = True):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.expand_dims(dst, axis = 2)
        dst = np.concatenate((dst, dst, dst), axis = 2)
    if scale:
        resize_h = scr.shape[0]
        resize_w = scr.shape[1]
        dst = cv2.resize(dst, (resize_w, resize_h))
    else:
        resize_h = dst.shape[0]
        resize_w = dst.shape[1]
        scr = cv2.resize(scr, (resize_w, resize_h))
    psnr = measure.compare_psnr(scr, dst)
    return psnr

# Compute the mean structural similarity index between two images
def SSIM(srcpath, dstpath, gray2rgb = False, scale = True, RGBinput = True):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.expand_dims(dst, axis = 2)
        dst = np.concatenate((dst, dst, dst), axis = 2)
    if scale:
        resize_h = scr.shape[0]
        resize_w = scr.shape[1]
        dst = cv2.resize(dst, (resize_w, resize_h))
    else:
        resize_h = dst.shape[0]
        resize_w = dst.shape[1]
        scr = cv2.resize(scr, (resize_w, resize_h))
    ssim = measure.compare_ssim(scr, dst, multichannel = RGBinput)
    return ssim

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if os.path.join(root, filespath)[-3:] == 'jpg':
                ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-3:] == 'jpg':
                ret.append(filespath)
    return ret
    
# read a txt expect EOF
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
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

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# Traditional indexes accuracy for dataset
def Dset_Acuuracy(refpath_imglist, basepath_imglist, gray2rgb = False, scale = True):
    # Define the list saving the accuracy
    nrmselist = []
    psnrlist = []
    ssimlist = []
    nrmseratio = 0
    psnrratio = 0
    ssimratio = 0

    # Compute the accuracy
    for i in range(len(refpath_imglist)):
        # Full imgpath
        refimgpath = refpath_imglist[i]
        imgpath = basepath_imglist[i]
        print(refimgpath)
        print(imgpath)
        # Compute the traditional indexes
        nrmse = NRMSE(refimgpath, imgpath, gray2rgb, scale)
        psnr = PSNR(refimgpath, imgpath, gray2rgb, scale)
        ssim = SSIM(refimgpath, imgpath, gray2rgb, scale)
        nrmselist.append(nrmse)
        psnrlist.append(psnr)
        ssimlist.append(ssim)
        nrmseratio = nrmseratio + nrmse
        psnrratio = psnrratio + psnr
        ssimratio = ssimratio + ssim
        print('The %dth image: nrmse: %f, psnr: %f, ssim: %f' % (i, nrmse, psnr, ssim))
    nrmseratio = nrmseratio / len(refpath_imglist)
    psnrratio = psnrratio / len(refpath_imglist)
    ssimratio = ssimratio / len(refpath_imglist)

    return nrmselist, psnrlist, ssimlist, nrmseratio, psnrratio, ssimratio
    
if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--refpath', type = str, default = '', help = 'define reference path')
    parser.add_argument('--basepath', type = str, default = '', help = 'define imgpath')
    parser.add_argument('--gray2rgb', type = bool, default = False, help = 'whether there is an input is grayscale')
    parser.add_argument('--scale', type = bool, default = True, help = 'if True, then align the resize based on refpath; if False, based on basepath')
    parser.add_argument('--savelist', type = bool, default = False, help = 'whether the results should be saved')
    opt = parser.parse_args()
    print(opt)

    # Read all names
    refpath_imglist = get_files(opt.refpath)
    basepath_imglist = get_files(opt.basepath)
    a = get_jpgs(opt.refpath)
    b = get_jpgs(opt.basepath)
    assert a == b, 'the two dataset contains unpaired images which is wrong'
    nrmselist, psnrlist, ssimlist, nrmseratio, psnrratio, ssimratio = Dset_Acuuracy(refpath_imglist, basepath_imglist, gray2rgb = opt.gray2rgb, scale = opt.scale)
    print('The overall results: nrmse: %f, psnr: %f, ssim: %f' % (nrmseratio, psnrratio, ssimratio))

    # Save the files
    if opt.savelist:
        text_save(nrmselist, "./nrmselist.txt")
        text_save(psnrlist, "./psnrlist.txt")
        text_save(ssimlist, "./ssimlist.txt")
    