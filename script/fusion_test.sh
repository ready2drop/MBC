#!/bin/bash


## early fusion
# python src/train/test_fusion.py --model_architecture 'ResNet' --fusion early --batch_size 50 --ckpt_path /home/rkdtjdals97/MBC/logs/2024-12-04-20-42-train-mm-ResNet/
# python src/train/test_fusion.py --model_architecture 'ViT' --fusion early --batch_size 50 --ckpt_path /home/rkdtjdals97/MBC/logs/2024-12-04-20-50-train-mm-ViT/


# ## intermediate fusion

# python src/train/test_fusion.py --model_architecture 'ResNet' --fusion intermediate --batch_size 20 --ckpt_path /home/rkdtjdals97/MBC/logs/2024-12-04-21-32-train-mm-ResNet/
# python src/train/test_fusion.py --model_architecture 'ViT' --fusion intermediate --batch_size 20 --ckpt_path /home/rkdtjdals97/MBC/logs/2024-12-04-21-38-train-mm-ViT/

# ## late fusion

python src/train/test_fusion.py --model_architecture 'ResNet' --fusion late --batch_size 20 --output_dim 1
python src/train/test_fusion.py --model_architecture 'ViT' --fusion late --batch_size 20 --output_dim 1