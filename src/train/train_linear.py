import sys
import os
from tqdm import tqdm
import numpy as np
import random
import wandb
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from torch.nn.parallel import DataParallel

from src.dataset.bc_dataloader import getloader_bc
from src.model.mbc import MultiModalbileductClassifier_2d, MultiModalbileductClassifier_3d
from src.model.ibc import ImagebileductClassifier_2d, ImagebileductClassifier_3d

from src.utils.loss import get_optimizer_loss_scheduler
from src.utils.util import logdir, get_model_parameters


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) 
    random.seed(seed) 
seed_everything(42)
    
# Training and Valaidation
class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, device, dict):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.data_path = dict['data_path']
        self.excel_file = dict['excel_file']
        self.batch_size = dict['batch_size']
        self.log_dir = dict['log_dir']
        self.epochs = dict['epochs']
        self.val_every = dict['val_every']
        self.mode = dict['mode']
        self.modality = dict['modality']
        self.use_wandb = dict['use_wandb']
        self.phase = dict['phase']

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        with tqdm(total=len(train_loader), desc="Training") as pbar: 
            if self.modality == 'mm':
                for images, features, targets, _ in train_loader:
                    self.optimizer.zero_grad()
                    # outputs = self.model(images.to(self.device), [feature.to(self.device) for feature in features])
                    outputs = self.model(images.to(self.device), features.to(self.device))
                    loss = self.loss_fn(outputs.squeeze(), targets.squeeze().float().to(self.device))  # Squeeze output and convert labels to float
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
                    total_train += targets.size(0)
                    correct_train += (predicted == targets.to(self.device)).sum().item()
                    pbar.update(1)
                    pbar.set_postfix({'Loss': f"{running_loss/pbar.n:.4f}", 'Accuracy': f"{correct_train/total_train:.4f}"})

                train_loss = running_loss / len(train_loader)
                train_acc = correct_train / total_train
                
            elif self.modality == 'image':
                for images, targets, _ in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(images.to(self.device))
                    loss = self.loss_fn(outputs.squeeze(), targets.squeeze().float().to(self.device))  # Squeeze output and convert labels to float
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
                    total_train += targets.size(0)
                    correct_train += (predicted == targets.to(self.device)).sum().item()
                    pbar.update(1)
                    pbar.set_postfix({'Loss': f"{running_loss/pbar.n:.4f}", 'Accuracy': f"{correct_train/total_train:.4f}"})

                train_loss = running_loss / len(train_loader)
                train_acc = correct_train / total_train    

        return train_loss, train_acc

    def train(self):
        train_loader, valid_loader = getloader_bc(self.data_path, self.excel_file, self.batch_size, self.mode, self.modality, self.phase)
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        best_val_acc = 0.0

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            if (epoch + 1) % self.val_every == 0:
                val_loss, val_acc = self.validate(valid_loader)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                if self.use_wandb:
                    wandb.log({"train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch+1)
                    
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

                # Save the best model based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print("Model Was Saved ! Current Best Accuracy: {} Current Accuracy: {}".format(best_val_acc, val_acc))
                    torch.save(self.model.module.state_dict(), f'{self.log_dir}/best_epoch_weights.pth')
                else: 
                    print("Model Was Not Saved ! Current Best Accuracy: {} Current Accuracy: {}".format(best_val_acc, val_acc))
                
                # Step the scheduler
                self.scheduler.step()

            

        return train_losses, val_losses, train_accs, val_accs

    def validate(self, valid_loader):
        self.model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            if self.modality == 'mm':
                for images, features, targets, _ in tqdm(valid_loader, desc="Validation"):
                    # outputs = self.model(images.to(self.device), [feature.to(self.device) for feature in features])
                    outputs = self.model(images.to(self.device), features.to(self.device))
                    loss = self.loss_fn(outputs.squeeze(), targets.squeeze().float().to(self.device))  # Squeeze output and convert labels to float
                    val_running_loss += loss.item()
                    predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
                    total_val += targets.size(0)
                    correct_val += (predicted == targets.to(self.device)).sum().item()
            elif self.modality == 'image':
                for images, targets, _ in tqdm(valid_loader, desc="Validation"):
                    outputs = self.model(images.to(self.device))
                    loss = self.loss_fn(outputs.squeeze(), targets.squeeze().float().to(self.device))  # Squeeze output and convert labels to float
                    val_running_loss += loss.item()
                    predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
                    total_val += targets.size(0)
                    correct_val += (predicted == targets.to(self.device)).sum().item()  

        val_loss = val_running_loss / len(valid_loader)
        val_acc = correct_val / total_val

        return val_loss, val_acc



device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

parser = argparse.ArgumentParser(description="Multimodal Bile duct stone Classfier")
parser.add_argument("--epochs", default=3000, type=int, help="Epoch")
parser.add_argument("--val_every", default=100, type=int, help="Learning rate")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--optimizer", default='adamw', type=str, help="Type of Optimizer") # 'adam', 'rmsprop'
parser.add_argument("--momentum", default=0.0, type=float, help="Add momentum for SGD optimizer")
parser.add_argument("--loss_function", default='BCE', type=str, help="Type of Loss function")
parser.add_argument("--scheduler", default='warmup_cosine', type=str, help="Type of Learning rate scheduler") # 'stepLR','CosineAnnealingLR'
parser.add_argument("--batch_size", default=20, type=int, help="Batch size")
parser.add_argument("--num_gpus", default="0,1", type=str, help="Comma-separated list of GPU numbers")
parser.add_argument("--num_classes", default=1, type=int, help="Assuming binary classification")
parser.add_argument("--use_parallel", action='store_true', help="Use Weights and Biases for logging")
parser.add_argument("--use_wandb", action='store_true', help="Use Weights and Biases for logging")
parser.add_argument("--model_architecture", default="efficientnet_b0", type=str, help="Model architecture")
parser.add_argument("--data_path", default=None, type=str, help="Directory of dataset")
parser.add_argument("--image_pretrain_path", default=None, type=str, help="pretrained weight path")
parser.add_argument("--excel_file", default='dumc_1024a.csv', type=str, help="tabular data")
parser.add_argument("--data_shape", default='3d', type=str, help="Input data shape") # '3d','2d'
parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")
parser.add_argument("--mode", default='train', type=str, help="mode") # 'train', 'test'
parser.add_argument("--modality", default='mm', type=str, help="modality") # 'mm', 'image', 'tabular'
parser.add_argument("--phase", default='combine', type=str, help="CT phase") # 'portal', 'pre-enhance', 'combine'

args = parser.parse_args()
args.log_dir = logdir(args.log_dir, args.mode, args.modality, args.model_architecture)

PARAMS = vars(args)
PARAMS = get_model_parameters(PARAMS)

if PARAMS['use_wandb'] == True:
    wandb.init(project="Multimodal-Bileductstone-Classifier", save_code=True, name = f"{PARAMS['model_architecture']},{PARAMS['modality']}, {PARAMS['data_shape']}", config=PARAMS)

# Modality, Model, Data Shape
if PARAMS['modality'] == 'mm':
    if PARAMS['data_shape'] == '3d':
        model = MultiModalbileductClassifier_3d(PARAMS)
    else:
        model = MultiModalbileductClassifier_2d(PARAMS)
elif PARAMS['modality'] == 'image':
    if PARAMS['data_shape'] == '3d':
        model = ImagebileductClassifier_3d(PARAMS)
    else:
        model = ImagebileductClassifier_2d(PARAMS)
else: 
    pass

# Model freeze    
# for param in model.parameters(): 
#     param.requires_grad = False
  
# for param in model.fc.parameters():
#     param.requires_grad = True
        
# Data parallel
if PARAMS['use_parallel']:
    # model = DataParallel(model, device_ids=[i for i in range(PARAMS['num_gpus'])]).to(device)
    model = DataParallel(model, device_ids=[int(gpu) for gpu in PARAMS['num_gpus'].split(",")] ).to(device)
else:
    model.to(device)

    
# loss, optimizer, scheduler
loss_fn, optimizer, scheduler = get_optimizer_loss_scheduler(PARAMS, model)


# Model train
trainer = Trainer(model, optimizer, scheduler, loss_fn, device, PARAMS)
train_losses, val_losses, train_accs, val_accs = trainer.train()

