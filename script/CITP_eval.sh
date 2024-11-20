#!/bin/bash


## Contrastive Image Tabular Pretraining
python src/train/CITP_evaluate.py --model_architecture 'ViT' --pretrained_dir '/home/rkdtjdals97/MBC/logs/2024-11-11-19-10-pretrain-mm' --batch_size 1 --hidden_dim 128 --use_parallel --use_wandb
# python src/train/CITP_evaluate.py --model_architecture 'ResNet' --pretrained_dir '/home/rkdtjdals97/MBC/logs/2024-11-11-19-08-pretrain-mm' --batch_size 1 --hidden_dim 128 --use_parallel --use_wandb
