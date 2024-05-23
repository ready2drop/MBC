#!/bin/bash

## Multimodel classification
## 2D
# python train.py --model_architecture 'efficientnet_v2_l' --epochs 1000 --data_shape '2d' --modality 'mm' --use_wandb
# python train.py --model_architecture 'convnext_large' --epochs 1000 --data_shape '2d' --modality 'mm' --use_wandb
# python train.py --model_architecture 'regnet_y_32gf'  --epochs 1000 --data_shape '2d' --modality 'mm' --use_wandb
# python train.py --model_architecture 'resnext101_64x4d' --epochs 1000 --data_shape '2d' --modality 'mm' --use_wandb
## 3D
# python train.py --model_architecture 'SwinUNETR' --epochs 3000 --val_every 100 --data_shape '3d' --modality 'mm' --data_path '/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti_crop/' --use_parallel --use_wandb 

## Image classification
## 3D
python train.py --model_architecture 'SwinUNETR' --epochs 3000 --val_every 100 --data_shape '3d' --modality 'image' --data_path '/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti_crop/' --use_parallel --use_wandb 
