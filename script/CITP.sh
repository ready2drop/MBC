#!/bin/bash


## Contrastive Image Tabular Pretraining
python src/train/CITP.py --model_architecture 'ViT' --num_gpus '5' --batch_size 88 --hidden_dim 128 --use_parallel --use_wandb
# python src/train/CITP.py --model_architecture 'ResNet' --num_gpus '4' --batch_size 8 --hidden_dim 128 --use_parallel --use_wandb
