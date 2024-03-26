#!/bin/bash
python train.py --model_architecture 'efficientnet_v2_l'
python train.py --model_architecture 'convnext_large'
python train.py --model_architecture 'regnet_y_32gf' 
python train.py --model_architecture 'resnext101_64x4d'
# python train.py --model_architecture 'vit_l_16'
