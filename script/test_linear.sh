#!/bin/bash

## Multimodel classification
## 2D
# python train.py --model_architecture 'efficientnet_v2_l' --epochs 1000
# python train.py --model_architecture 'convnext_large' --epochs 1000
# python train.py --model_architecture 'regnet_y_32gf'  --epochs 1000
# python train.py --model_architecture 'resnext101_64x4d' --epochs 1000

## 3D

echo "test shell"

# python src/train/test_linear.py --model_architecture ViT --ckpt_path /home/rkdtjdals97/MBC/logs/2024-11-15-16-59-train-mm/best_epoch_weights.pth --data_path '/home/rkdtjdals97/datasets/DUMC_nifti_crop/' --modality mm --use_parallel --excel_file dumc_1024a.csv --use_wandb

# Wait for 50 seconds
# sleep 50

## Image classification
## 3D

# echo "test shell2"
python src/train/test_linear.py --model_architecture ViT --ckpt_path /home/rkdtjdals97/MBC/logs/2024-11-15-16-59-train-image/best_epoch_weights.pth --data_path '/home/rkdtjdals97/datasets/DUMC_nifti_crop/' --modality 'image'  --use_parallel --excel_file dumc_1024a.csv --use_wandb

