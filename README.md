# MBC: Multimodal Bile Duct Stone Classifier

This repository contains code for training and testing a Multimodal Bile Duct Stone Classifier using PyTorch.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Arguments](#arguments)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [References](#references)

## Introduction

This classifier is designed to classify bile duct stone images using a multimodal approach, combining different types of medical images. The model architecture used is based on the EfficientNet-B0 architecture, and training is performed using a binary cross-entropy loss function. Testing evaluates the trained model on a separate test set and provides accuracy and loss metrics.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- wandb
- tqdm

You can install the required packages using the following command:
'''
pip install -r requirements.txt
'''

## Usage

### Training
You can run the training script using the following command:
'''
python train.py --epochs 100 --val_every 10 --learning_rate 0.001 --batch_size 16 --num_gpus 8 --optimizer adam --loss_function BCE --scheduler StepLR --momentum 0.0 --model_architecture efficientnet_b0 --data_path /path/to/dataset/ --pretrain_path /path/to/pretrained_weights.pt --excel_file bileduct_data.xlsx --data_shape 3d --log_dir logs/ --mode train
'''
### Testing
To evaluate the trained model, use the following command:
'''
python test.py --epochs 100 --learning_rate 0.001 --batch_size 1 --num_gpus 8 --num_classes 1 --optimizer adam --loss_function BCE --scheduler StepLR --momentum 0.0 --model_architecture efficientnet_b0 --data_path /path/to/dataset/ --pretrain_path /path/to/pretrained_weights.pt --ckpt_path /path/to/best_epoch_weights.pth --excel_file bileduct_data.xlsx --data_shape 3d --log_dir logs/ --mode test
'''

## Arguments

- `--epochs`: Number of epochs for training/testing.
- `--val_every`: Number of epochs to validate after during training.
- `--learning_rate`: Learning rate for the optimizer.
- `--batch_size`: Batch size for training/testing.
- `--num_gpus`: Number of GPUs to use for training/testing.
- `--num_classes`: Number of classes for classification.
- `--optimizer`: Type of optimizer to use (adam, rmsprop).
- `--loss_function`: Type of loss function to use (BCE, ...).
- `--scheduler`: Type of learning rate scheduler to use (StepLR, CosineAnnealingLR).
- `--momentum`: Momentum for SGD optimizer.
- `--model_architecture`: Model architecture to use (efficientnet_b0, ...).
- `--data_path`: Directory containing the dataset.
- `--pretrain_path`: Path to pre-trained weights.
- `--ckpt_path`: Path to the best epoch weights for testing.
- `--excel_file`: Excel file containing dataset information.
- `--data_shape`: Input data shape (2d, 3d).
- `--log_dir`: Directory to save logs.
- `--mode`: Mode of operation (train, test).

## Training

During training, the script will output the training loss, training accuracy, validation loss, and validation accuracy for each epoch. The best model based on validation accuracy will be saved to the specified log directory.

## Testing

After training, the script will evaluate the trained model on a separate test set and provide accuracy and loss metrics. Additionally, it will save the confusion matrix and ROC curve plots in the log directory.

## Results

The results of training and testing, including loss curves, accuracy metrics, confusion matrix, and ROC curves, will be saved in the log directory.

## References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [wandb Documentation](https://docs.wandb.ai/)
