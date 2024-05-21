#!/bin/bash
# python train.py --model_architecture 'efficientnet_v2_l' --epochs 1000
# python train.py --model_architecture 'convnext_large' --epochs 1000
# python train.py --model_architecture 'regnet_y_32gf'  --epochs 1000
# python train.py --model_architecture 'resnext101_64x4d' --epochs 1000
python test.py --model_architecture 'SwinUNETR' --ckpt_path /home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-05-21-15-53-train/best_epoch_weights.pth --data_path '/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti_crop/' --data_shape '3d' --modality 'image' --mode 'test' --use_parallel 

