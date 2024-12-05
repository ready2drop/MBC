
#!/bin/bash


# early fusion
python src/train/train_fusion.py --model_architecture 'ResNet' --fusion early --batch_size 100 
python src/train/train_fusion.py --model_architecture 'ViT' --fusion early --batch_size 100


# ## intermediate fusion

# python src/train/train_fusion.py --model_architecture 'ResNet' --fusion intermediate --batch_size 20 
# python src/train/train_fusion.py --model_architecture 'ViT' --fusion intermediate --batch_size 20

# ## late fusion

# python src/train/train_fusion.py --model_architecture 'ResNet' --fusion late --batch_size 20 --output_dim 1 
# python src/train/train_fusion.py --model_architecture 'ViT' --fusion late --batch_size 20 --output_dim 1