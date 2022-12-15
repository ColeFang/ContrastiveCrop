## Mixup Crop on contrastive learning


This is the official PyTorch implementation:

This repo includes PyTorch implementation of **SimCLR**, **MoCo**, **BYOL** and **SimSiam**, as well as their DDP training code.
## Preparation
1. Create a python enviroment with `pytorch >= 1.8.1`.
2. `pip install -r requirements.txt`


## Datasets
Please download and organize the datasets in this structure:
```
ContrastiveCrop
├── data/
    ├── ImageNet/
    │   ├── train/ 
    │   ├── val/
    ├── cifar-10-batches-py/
    ├── cifar-100-python/
    ├── stl10_binary/
    ├── tiny-imagenet-200/
    │   ├── train/
    │   ├── val/
```
Use this [script](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4) to prepare tiny-imagenet-200.

## Pre-train
```
# MoCo, CIFAR-10, CCrop
python DDP_moco_ccrop.py configs/small/cifar10/moco_ccrop.py

# SimSiam, CIFAR-100, MCrop
python DDP_simsiam_mcrop.py configs/small/cifar100/simsiam_mcrop.py

# MoCo V2, IN-200, CCrop
python DDP_moco_ccrop.py configs/IN200/mocov2_ccrop.py

# MoCo V2, IN-1K, CCrop
python DDP_moco_ccrop.py configs/IN1K/mocov2_ccrop.py
```
We also recommend trying an even simpler version of ContrastiveCrop, named **SimCCrop**, 
that simply fixes a box at the center of the image with half height & width of that image.
**SimCCrop** even does not require localization and thus adds **NO** extra training overhead.
It should work well on almost 'object-centric' datasets.
```
# MoCo, SimCCrop
python DDP_moco_ccrop.py configs/small/cifar10/moco_simccrop.py
python DDP_moco_ccrop.py configs/small/cifar100/moco_simccrop.py
```

## Linear Evaluation
```
# CIFAR-10
python DDP_linear.py configs/linear/cifar10_res18.py --load ./checkpoints/small/cifar10/moco_ccrop/last.pth

# CIFAR-100
python DDP_linear.py configs/linear/cifar100_res18.py --load ./checkpoints/small/cifar100/simsiam_mcrop/last.pth

# IN-200 
python DDP_linear.py configs/linear/IN200_res50.py --load ./checkpoints/IN200/mocov2_ccrop/last.pth

# IN-1K
python DDP_linear.py configs/linear/IN1K_res50.py --load ./checkpoints/IN1K/mocov2_ccrop/last.pth
```

More models and datasets coming soon.
