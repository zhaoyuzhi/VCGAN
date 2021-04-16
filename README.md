# VCGAN

VCGAN is a hybrid colorization model that works for both image and video colorization tasks.

## 1 Pre-requisite

Note that this project is implemented using Python 3.6, CUDA 8.0, and PyTorch 1.0.0 (minimum requirement).

Besides, the `cupy`, `opencv`, and `scikit-image` libs are used for this project.

Please build an appropriate environment for the [PWC-Net](https://github.com/NVlabs/PWC-Net) to compute optical flow.

## 2 Visual comparisons of VCGAN and the other SOTA video colorization methods

So far we visualize two samples included in the main paper for better visualization.

### 2.1 DAVIS dataset - 'horsejump-high'

The representative image is shown as (please compare the significant regions marked by red rectangles):

<img src="./img/main1.png" />

The gifs are shown as:

| Column 1 | Column 2 | Column 3 | Column 4 |
|  ----  | ----  |  ----  | ----  |
| CIC | CIC+BTC | LTBC | LTBC+BTC |
| SCGAN | SCGAN+BTC | ChromaGAN | ChromaGAN+BTC |
| 3DVC | FAVC | VCGAN | |
| Grayscale | Ground Truth | | |

<img src="./img/horsejump-high/CIC.gif" width="238" height="135" /><img src="./img/horsejump-high/CIC+BTC.gif" width="238" height="135" /><img src="./img/horsejump-high/LTBC.gif" width="238" height="135" /><img src="./img/horsejump-high/LTBC+BTC.gif" width="238" height="135" />

<img src="./img/horsejump-high/SCGAN.gif" width="238" height="135" /><img src="./img/horsejump-high/SCGAN+BTC.gif" width="238" height="135" /><img src="./img/horsejump-high/ChromaGAN.gif" width="238" height="135" /><img src="./img/horsejump-high/ChromaGAN+BTC.gif" width="238" height="135" />

<img src="./img/horsejump-high/3DVC.gif" width="238" height="135" /><img src="./img/horsejump-high/FAVC.gif" width="238" height="135" /><img src="./img/horsejump-high/VCGAN.gif" width="238" height="135" />

<img src="./img/horsejump-high/Grayscale.gif" width="238" height="135" /><img src="./img/horsejump-high/GT.gif" width="238" height="135" />

### 2.2 Videvo dataset - 'SkateboarderTableJump'

The representative image is shown as (please compare the significant regions marked by red rectangles):

<img src="./img/main2.png" />

The gifs are shown as:

| Column 1 | Column 2 | Column 3 | Column 4 |
|  ----  | ----  |  ----  | ----  |
| CIC | CIC+BTC | LTBC | LTBC+BTC |
| SCGAN | SCGAN+BTC | ChromaGAN | ChromaGAN+BTC |
| 3DVC | FAVC | VCGAN | |
| Grayscale | Ground Truth | | |

<img src="./img/SkateboarderTableJump/CIC.gif" width="238" height="135" /><img src="./img/SkateboarderTableJump/CIC+BTC.gif" width="238" height="135" /><img src="./img/SkateboarderTableJump/LTBC.gif" width="238" height="135" /><img src="./img/SkateboarderTableJump/LTBC+BTC.gif" width="238" height="135" />

<img src="./img/SkateboarderTableJump/SCGAN.gif" width="238" height="135" /><img src="./img/SkateboarderTableJump/SCGAN+BTC.gif" width="238" height="135" /><img src="./img/SkateboarderTableJump/ChromaGAN.gif" width="238" height="135" /><img src="./img/SkateboarderTableJump/ChromaGAN+BTC.gif" width="238" height="135" />

<img src="./img/SkateboarderTableJump/3DVC.gif" width="238" height="135" /><img src="./img/SkateboarderTableJump/FAVC.gif" width="238" height="135" /><img src="./img/SkateboarderTableJump/VCGAN.gif" width="238" height="135" />

<img src="./img/SkateboarderTableJump/Grayscale.gif" width="238" height="135" /><img src="./img/SkateboarderTableJump/GT.gif" width="238" height="135" />

## 3 Visual comparisons of VCGAN and the other SOTA image colorization methods

We visualize the Figure 11 in the main paper for better view.

<img src="./img/image.png" />

## 4 Download pre-trained model

Please download pre-trained ResNet50-Instance-Normalized model at this [link](https://portland-my.sharepoint.com/:u:/g/personal/yzzhao2-c_my_cityu_edu_hk/EXtVpiqYqMNIoQhqJaRrMM8BP8Zy4ZIDhscDhb0FhyAbPg?e=T4s0ha), if you want to train VCGAN. The hyper-parameters follow the settings of original paper except normalization.

Please download at this [link]() (will available after published), if you want to test VCGAN. Then put this model in the `models` folder under current path.

## 5 Use the code

Put the pre-trained ResNet50-Instance-Normalized model into `trained_models` folder, then change the settings and train VCGAN in first stage:

```bash
cd train
python train.py or sh first.sh
```

After the model is trained, you can run following codes for second stage:

```bash
python train2.py or sh second.sh (on 256p resolution)
python train2.py or sh third.sh (on 480p resolution)
```

For testing, please run (note that you need to change path to models):

```bash
python test_model_*.py
```

## 6 Related Projects

**SCGAN (IEEE TCSVT 2020): [Project](https://github.com/zhaoyuzhi/Semantic-Colorization-GAN)
[Paper](https://ieeexplore.ieee.org/document/9257445/keywords#keywords)
[Github](https://github.com/zhaoyuzhi/Semantic-Colorization-GAN)**

**ChromaGAN (WACV 2020): [Project](https://junyanz.github.io/CycleGAN/)
[Paper](https://openaccess.thecvf.com/content_WACV_2020/html/Vitoria_ChromaGAN_Adversarial_Picture_Colorization_with_Semantic_Class_Distribution_WACV_2020_paper.html)
[Github](https://junyanz.github.io/CycleGAN/)**

**Learning Representations for Automatic Colorization (ECCV 2016): [Project](http://people.cs.uchicago.edu/~larsson/colorization/)
[Paper](https://arxiv.org/abs/1603.06668)
[Github](https://github.com/gustavla/autocolorize)**

**Colorful Image Colorization (ECCV 2016): [Project](http://richzhang.github.io/colorization/)
[Paper](https://arxiv.org/abs/1603.08511)
[Github](https://github.com/richzhang/colorization)**

**Let there be Color! (ACM TOG 2016): [Project](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/)
[Paper](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf)
[Github](https://github.com/satoshiiizuka/siggraph2016_colorization)**

**Pix2Pix (CVPR 2017): [Project](https://phillipi.github.io/pix2pix/)
[Paper](https://arxiv.org/pdf/1611.07004.pdf)
[Github](https://github.com/phillipi/pix2pix)**

**CycleGAN (ICCV 2017): [Project](https://junyanz.github.io/CycleGAN/)
[Paper](https://arxiv.org/pdf/1703.10593.pdf)
[Github](https://github.com/junyanz/CycleGAN)**
