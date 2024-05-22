#!/bin/bash

## Multimodel classification
## 2D
# python train.py --model_architecture 'efficientnet_v2_l' --epochs 1000
# python train.py --model_architecture 'convnext_large' --epochs 1000
# python train.py --model_architecture 'regnet_y_32gf'  --epochs 1000
# python train.py --model_architecture 'resnext101_64x4d' --epochs 1000

## 3D
python test.py --model_architecture 'SwinUNETR' --data_path '/home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-05-22-00-14-train/best_epoch_weights.pth' --data_path '/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti_crop/' --data_shape '3d' --modality 'mm' --use_parallel --use_wandb

# Wait for 50 seconds
sleep 50

## Image classification
## 3D
python test.py --model_architecture 'SwinUNETR' --ckpt_path /home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-05-21-15-53-train/best_epoch_weights.pth --data_path '/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti_crop/' --data_shape '3d' --modality 'image'  --use_parallel --use_wandb

