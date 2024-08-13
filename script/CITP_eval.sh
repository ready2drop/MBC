#!/bin/bash


## Contrastive Image Tabular Pretraining
python src/train/CITP_evaluate.py --model_architecture 'ViT' --pretrained_dir '/home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-08-02-16-58-pretrain-mm' --batch_size 1 --hidden_dim 512 --use_parallel --use_wandb
# python src/train/CITP_evaluate.py --model_architecture 'ResNet' --epochs 1000 --batch_size 16 --hidden_dim 64 --use_parallel --use_wandb
