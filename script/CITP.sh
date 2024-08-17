#!/bin/bash


## Contrastive Image Tabular Pretraining
# python src/train/CITP.py --model_architecture 'ViT' --epochs 1000 --batch_size 88 --hidden_dim 64 --use_parallel --use_wandb
python src/train/CITP.py --model_architecture 'ResNet' --epochs 1000 --batch_size 16 --hidden_dim 128 --use_parallel --use_wandb
