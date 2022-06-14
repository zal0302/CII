## Rethinking the U-shape Structure for Salient Object Detection

### This is a PyTorch implementation of our TIP 2021 [paper](https://mftp.mmcheng.net/Papers/21TIP-CII.pdf).

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

Download the following pre-trained model for [CII](https://drive.google.com/file/d/1JcePr4FwWMedhFHeClYF1v_MIYwJGOF0/view?usp=sharing) into `saved/models` folder. 

Download the following pre-trained model for [backbone](https://drive.google.com/file/d/1zZJaB5-SnRuKuITmVkWNWQvRHJrOymmw/view?usp=sharing) into `pretrained` folder. 

### 4. Test

For all datasets testing used in our paper

```shell
python test.py -r saved/models/cii.pth
```

All results saliency maps will be stored under `saved/results` folders in .png formats.

### 5. Pre-computed results and evaluation results

You may refer to this repo for results evaluation: [SalMetric](https://github.com/Andrew-Qibin/SalMetric).

We provide the [pre-computed saliency maps and evaluation results](https://drive.google.com/file/d/11Uj2-qNDyASrfvdXj2uE9Zm7xiYYwNEM/view?usp=sharing).

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
