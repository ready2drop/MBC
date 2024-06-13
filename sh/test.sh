#!/bin/bash

## Multimodel classification
## 2D
# python train.py --model_architecture 'efficientnet_v2_l' --epochs 1000
# python train.py --model_architecture 'convnext_large' --epochs 1000
# python train.py --model_architecture 'regnet_y_32gf'  --epochs 1000
# python train.py --model_architecture 'resnext101_64x4d' --epochs 1000

## 3D
python test.py --model_architecture SwinUNETR --ckpt_path /home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-06-08-15-11-train-mm/best_epoch_weights.pth --data_path '/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti_crop/' --data_shape 3d --modality mm --use_parallel --excel_file dumc_0603.csv

# Wait for 50 seconds
# sleep 50

## Image classification
## 3D
# python test.py --model_architecture 'SwinUNETR' --ckpt_path /home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-05-23-19-05-train-image/best_epoch_weights.pth --data_path '/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti_crop/' --data_shape '3d' --modality 'image'  --use_parallel 

