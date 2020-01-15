# Pytorch Implemented Deep-SR-ITM
A Pytorch implemented Deep SR-ITM (ICCV2019 oral)

> Soo Ye Kim, Munchurl Kim. A Multi-purpose Convolutional Neural Network for Simultaneous Super-Resolution and High Dynamic Range Image Reconstruction. In Proceedings of Asian Conference on Computer Vision, 2018.

<b>Note: </b> The code is completely adapted from [`https://github.com/sooyekim/Deep-SR-ITM`](https://github.com/sooyekim/Deep-SR-ITM) but rewritten in pytorch format. This repository is NOT aimed to improve the baseline, but to retain the original settings in a different implementation. If you have any questions for the details of the implementations, please refer to the original repo. 

## Test Environment
* Ubuntu 16.04 LTS
* python 3.7.5
* pytorch 1.3.1
* torchvision 0.4.2
* CUDA 10.1
* opencv 3.4.2
* numpy 1.17.4

## Data Preparation
1. Download training and testing data from [`https://github.com/sooyekim/Deep-SR-ITM`](https://github.com/sooyekim/Deep-SR-ITM)
2. Use Matlab to transform the data with extension, `.mat`, into `'.png'` form (No matter SDR or HDR images)
3. Prepare the data as following...
```
${DATA_ROOT}
├── trainset_SDR
│   ├── 000001.png
│   ├── 000002.png
│   ├── ...
│   └── 039840.png
├── trainset_HDR
│   ├── 000001.png
│   ├── 000002.png
│   ├── ...
│   └── 039840.png
├── testset_SDR
│   ├── 000001.png
│   ├── 000002.png
│   ├── ...
│   └── 000028.png
└── testset_SDR
    ├── 000001.png
    ├── 000002.png
    ├── ...
    └── 000028.png
```

## Prepare Environment
```
# Prepare CUDA Installation
...

# git clone repository
git clone https://github.com/greatwallet/Pytorch-Implemented-Deep-SR-ITM.git
cd Pytorch-Implemented-Deep-SR-ITM

# create conda environment
conda create --n env-sr-itm python=3.7 -y
conda activate env-sr-itm
conda install -c pytorch pytorch -y
conda install -c pytorch torchvision -y
conda install -c conda-forge opencv -y
conda install numpy -y

# set soft link to data path
ln -s ${DATA_ROOT} ./
```

## Usage
The default parameters in the scripts is set strictly according to the original repo. However, please modify the parameters in the script if you would like. 
### Train
```
python train_base_net.py
python train_full_net.py
```
### Test
Please specify the path of the testset and other settings in the [`test.py`](https://github.com/greatwallet/Pytorch-Implemented-Deep-SR-ITM/blob/master/test.py#L24).
```
python test.py
```

<b>Note: </b> The difference between the `val` and `test` phase in [`YouTubeDataset.__init__`](https://github.com/greatwallet/Pytorch-Implemented-Deep-SR-ITM/blob/master/dataset.py#L20) would be that:
- `val`: the SDR and HDR images must be both be provided, and the size of SDR image must be IDENTICAL with HDR images, and the YouTubeDataset will resize the SDR images for the net later on. 
- `test`: the HDR images may or may not be provided to the dataset. If provided, the size of SDR image should be `k` times smaller than HDR images (assuming `k` is the parameter `scale` of networks)

## Acknowledgement
SSIM and MS-SSIM functions are borrowed from [`https://github.com/VainF/pytorch-msssim`](https://github.com/VainF/pytorch-msssim)

## Contact
Please contact me via email (cxt_tsinghua@126.com) for any problems regarding the released code.
