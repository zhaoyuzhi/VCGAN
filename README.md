# VCGAN

VCGAN is a hybrid colorization model that works for both image and video colorization tasks.

link: https://ieeexplore.ieee.org/document/9721653

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

<img src="./img/horsejump-high/CIC.gif" width="200" height="117" /><img src="./img/horsejump-high/CIC+BTC.gif" width="200" height="117" /><img src="./img/horsejump-high/LTBC.gif" width="200" height="117" /><img src="./img/horsejump-high/LTBC+BTC.gif" width="200" height="117" />

<img src="./img/horsejump-high/SCGAN.gif" width="200" height="117" /><img src="./img/horsejump-high/SCGAN+BTC.gif" width="200" height="117" /><img src="./img/horsejump-high/ChromaGAN.gif" width="200" height="117" /><img src="./img/horsejump-high/ChromaGAN+BTC.gif" width="200" height="117" />

<img src="./img/horsejump-high/3DVC.gif" width="200" height="117" /><img src="./img/horsejump-high/FAVC.gif" width="200" height="117" /><img src="./img/horsejump-high/VCGAN.gif" width="200" height="117" />

<img src="./img/horsejump-high/Grayscale.gif" width="200" height="117" /><img src="./img/horsejump-high/GT.gif" width="200" height="117" />

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

<img src="./img/SkateboarderTableJump/CIC.gif" width="200" height="117" /><img src="./img/SkateboarderTableJump/CIC+BTC.gif" width="200" height="117" /><img src="./img/SkateboarderTableJump/LTBC.gif" width="200" height="117" /><img src="./img/SkateboarderTableJump/LTBC+BTC.gif" width="200" height="117" />

<img src="./img/SkateboarderTableJump/SCGAN.gif" width="200" height="117" /><img src="./img/SkateboarderTableJump/SCGAN+BTC.gif" width="200" height="117" /><img src="./img/SkateboarderTableJump/ChromaGAN.gif" width="200" height="117" /><img src="./img/SkateboarderTableJump/ChromaGAN+BTC.gif" width="200" height="117" />

<img src="./img/SkateboarderTableJump/3DVC.gif" width="200" height="117" /><img src="./img/SkateboarderTableJump/FAVC.gif" width="200" height="117" /><img src="./img/SkateboarderTableJump/VCGAN.gif" width="200" height="117" />

<img src="./img/SkateboarderTableJump/Grayscale.gif" width="200" height="117" /><img src="./img/SkateboarderTableJump/GT.gif" width="200" height="117" />

## 3 Visual comparisons of VCGAN and the other SOTA image colorization methods

We visualize the Figure 13 in the main paper for better view.

<img src="./img/image.png" />

## 4 Download pre-trained model

### 4.1 Training

Please download pre-trained ResNet50-Instance-Normalized model and other pre-trained models at this:

link: https://pan.baidu.com/s/1UH_MyLwCKPXyQag3k57RZg?pwd=d2fn code: d2fn

It contains a folder named `trained_models`; then you can put it under `train` folder. The hyper-parameters follow the settings of original paper except normalization.

### 4.1 Testing

Please download pre-trained VCGAN models at this:

link: https://pan.baidu.com/s/1jp2aUUYFT4A81ArerIsjyQ?pwd=j3gy code: j3gy

It contains a folder named `models`; then you can put it under `train` folder. Note that there are three models provided. The `model1_Second_Stage_epoch500_bs1_448p.pth` is by default.

Note that they are re-trained on a single GPU that might lead to slightly different results compared with the original one.

## 5 Use the code

Enter the `train` folder:

```bash
cd train
```

### 5.1 Training code

Put the pre-trained ResNet50-Instance-Normalized model into `trained_models` folder, then change the settings and train VCGAN in first stage:

```bash
python train.py or sh first.sh # on 256x256 image resolution
```

After the model is trained, you can run following codes for second stage:

```bash
python train2.py or sh second.sh # on 256p video resolution
python train2.py or sh third.sh # on 480p video resolution
```

### 5.2 Testing code

For testing, please run (note that you need to change path to models):

```bash
python test_model_second_stage_by_txt.py # for DAVIS dataset
python test_model_second_stage_by_txt2.py # for videvo dataset
python test_model_second_stage_by_folder.py # for a single folder
```

The network interpolation can also be used when applying different models:

```bash
python network_interp.py
```

## 6 Related Projects

**SCGAN: Saliency Map-guided Colorization with Generative Adversarial Network (IEEE TCSVT 2020): [Project](https://github.com/zhaoyuzhi/Semantic-Colorization-GAN)
[Paper](https://ieeexplore.ieee.org/document/9257445/keywords#keywords)
[Github](https://github.com/zhaoyuzhi/Semantic-Colorization-GAN)**

**ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution (WACV 2020):
[Paper](https://openaccess.thecvf.com/content_WACV_2020/html/Vitoria_ChromaGAN_Adversarial_Picture_Colorization_with_Semantic_Class_Distribution_WACV_2020_paper.html)
[Github](https://github.com/pvitoria/ChromaGAN)**

**FAVC: Fully Automatic Video Colorization With Self-Regularization and Diversity (CVPR 2019): [Project](https://leichenyang.weebly.com/project-color.html/)
[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lei_Fully_Automatic_Video_Colorization_With_Self-Regularization_and_Diversity_CVPR_2019_paper.pdf)
[Github](https://github.com/ChenyangLEI/automatic-video-colorization)**

**3DVC: Automatic Video Colorization using 3D Conditional Generative Adversarial Networks (ISVC 2019): [Paper](https://arxiv.org/abs/1905.03023?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)**

**BTC: Learning Blind Video Temporal Consistency (ECCV 2018): [Project](http://vllab.ucmerced.edu/wlai24/video_consistency/)
[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wei-Sheng_Lai_Real-Time_Blind_Video_ECCV_2018_paper.pdf)
[Github](https://github.com/phoenix104104/fast_blind_video_consistency)**

**LRAC: Learning Representations for Automatic Colorization (ECCV 2016): [Project](http://people.cs.uchicago.edu/~larsson/colorization/)
[Paper](https://arxiv.org/abs/1603.06668)
[Github](https://github.com/gustavla/autocolorize)**

**CIC: Colorful Image Colorization (ECCV 2016): [Project](http://richzhang.github.io/colorization/)
[Paper](https://arxiv.org/abs/1603.08511)
[Github](https://github.com/richzhang/colorization)**

**LTBC: Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification (ACM TOG 2016): [Project](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/)
[Paper](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf)
[Github](https://github.com/satoshiiizuka/siggraph2016_colorization)**

**Pix2Pix: Image-to-Image Translation with Conditional Adversarial Nets (CVPR 2017): [Project](https://phillipi.github.io/pix2pix/)
[Paper](https://arxiv.org/pdf/1611.07004.pdf)
[Github](https://github.com/phillipi/pix2pix)**

**CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (ICCV 2017): [Project](https://junyanz.github.io/CycleGAN/)
[Paper](https://arxiv.org/pdf/1703.10593.pdf)
[Github](https://github.com/junyanz/CycleGAN)**
