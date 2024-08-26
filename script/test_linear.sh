#!/bin/bash

## Multimodel classification
## 2D
# python train.py --model_architecture 'efficientnet_v2_l' --epochs 1000
# python train.py --model_architecture 'convnext_large' --epochs 1000
# python train.py --model_architecture 'regnet_y_32gf'  --epochs 1000
# python train.py --model_architecture 'resnext101_64x4d' --epochs 1000

## 3D
python src/train/test_linear.py --model_architecture SwinUNETR --ckpt_path /home/rkdtjdals97/MBC/logs/2024-08-17-18-29-train-image/best_epoch_weights.pth --data_path '/home/rkdtjdals97/datasets/Part5_nifti_crop/' --data_shape 3d --modality mm --use_parallel --excel_file dumc_0730a.csv --use_wandb

# Wait for 50 seconds
# sleep 50

## Image classification
## 3D
python src/train/test_linear.py --model_architecture 'SwinUNETR' --ckpt_path /home/rkdtjdals97/MBC/logs/2024-08-18-23-22-train-mm/best_epoch_weights.pth --data_path '/home/rkdtjdals97/datasets/Part5_nifti_crop/' --data_shape '3d' --modality 'image'  --use_parallel --excel_file dumc_0730a.csv --use_wandb

