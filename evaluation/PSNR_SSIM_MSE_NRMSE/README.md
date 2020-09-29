# Simple metrics: PSNR, SSIM, MSE, NRMSE

## 1 Pre-requisite

Note that the programs rely on opencv-python and scikit-image libs.

All the experiments are based on the original resolution of DAVIS and Videvo databases.

## 2 Resize

That means, the frames of generated 480p video should be resized to match the original size, which does not necessarily equal to 480p. (Many generated images are 448 * 832 actually).

## 3 Use the code

Please run this python file and provide your two paths to reference and target testing data, accompanied with a txt containing all the files' name.

```bash
python NRMSE_PSNR_SSIM_bylistname_align.py
```
