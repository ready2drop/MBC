#!/bin/bash


## Contrastive Image Tabular Pretraining
python src/train/CITP.py --model_architecture 'ViT' --epochs 1000 --batch_size 88 --hidden_dim 256 --use_parallel --use_wandb
