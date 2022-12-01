## Rethinking the U-shape Structure for Salient Object Detection

### This is the official PyTorch implementation of our TIP 2021 [paper](https://mftp.mmcheng.net/Papers/21TIP-CII.pdf).

## Prerequisites

- [Pytorch 0.4.1+](http://pytorch.org/)
- [torchvision](http://pytorch.org/)


## Usage

### 1. Clone the repository

```shell
git clone https://github.com/zal0302/CII.git
cd CII/
```

### 2. Download the datasets

Download the following [datasets for testing](https://drive.google.com/file/d/1jIL3Yvly4l4l_OggljjreD87pJIm_6rm/view?usp=sharing) and unzip them into `data` folder.

### 3. Download the pre-trained models for CII and backbone

Download the following pre-trained models for CII with [ResNet50 backbone](https://drive.google.com/file/d/1JcePr4FwWMedhFHeClYF1v_MIYwJGOF0/view?usp=sharing) and [ResNet18 backbone](https://drive.google.com/file/d/1DL860taDrmDUv-Am49AQZsdcF4Ey2-2t/view?usp=sharing) into `saved/models` folder. 

### 4. Test

For all datasets testing used in our paper for ResNet50 backbone:

```shell
python test.py -r saved/models/cii.pth -c saved/models/config.json
```

and for ResNet18 backbone:

```shell
python test.py -r saved/models/cii_res18.pth -c saved/models/config_resnet18.json
```

All results saliency maps will be stored under `saved/results` folders in .png formats.

### 5. Pre-computed results and evaluation results

You may refer to this repo for results evaluation: [SalMetric](https://github.com/Andrew-Qibin/SalMetric).

We provide the pre-computed saliency maps and evaluation results for [ResNet50 backbone](https://drive.google.com/file/d/11Uj2-qNDyASrfvdXj2uE9Zm7xiYYwNEM/view?usp=sharing) and [ResNet18 backbone](https://drive.google.com/file/d/1Q53oKWTNA9KznWmbXGm2IhY_2yeYvF1E/view?usp=sharing).

### 6. Contact

If you have any questions, feel free to contact me via: `liuzhiang(at)mail.nankai.edu.cn`.


### If you think this work is helpful, please cite

```latex
@article{liu2021rethinking,
  title={Rethinking the U-Shape Structure for Salient Object Detection},
  author={Liu, Jiang-Jiang and Liu, Zhi-Ang and Peng, Pai and Cheng, Ming-Ming},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={9030--9042},
  year={2021},
  publisher={IEEE}
}
```
```latex
@article{liu2022poolnet+,
  title={Poolnet+: Exploring the potential of pooling for salient object detection},
  author={Liu, Jiang-Jiang and Hou, Qibin and Liu, Zhi-Ang and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
