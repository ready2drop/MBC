import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from dataset.dataloader import getloader
from dataset.bc_dataloader import getloader_bc
from model.mbc import MultiModalbileductClassifier_2d, MultiModalbileductClassifier_3d
from utils.loss import get_optimizer_loss_scheduler
from utils.util import logdir, get_model_parameters

from timeit import default_timer as timer
from tqdm import tqdm
import os
import wandb
import argparse
import warnings
warnings.filterwarnings('ignore')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

parser = argparse.ArgumentParser(description="Multimodal Bile duct stone Classfier")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_gpus", default=8, type=int, help="Number of GPUs")
parser.add_argument("--num_classes", default=1, type=int, help="Assuming binary classification")
parser.add_argument("--optimizer", default='adam', type=str, help="Type of Optimizer") # 'adam', 'rmsprop'
parser.add_argument("--loss_function", default='BCE', type=str, help="Type of Loss function")
parser.add_argument("--scheduler", default='StepLR', type=str, help="Type of Learning rate scheduler") # 'stepLR','CosineAnnealingLR'
parser.add_argument("--momentum", default=0.0, type=float, help="Add momentum for SGD optimizer")
parser.add_argument("--model_architecture", default="efficientnet_b0", type=str, help="Model architecture")
parser.add_argument("--data_path", default='/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti/', type=str, help="Directory of data")
parser.add_argument("--pretrain_path", default='/home/irteam/rkdtjdals97-dcloud-dir/model_swinvit.pt', type=str, help="pretrained weight path")
parser.add_argument("--excel_file", default='bileduct_data_20240508b.xlsx', type=str, help="excel data")
parser.add_argument("--data_shape", default='3d', type=str, help="Input data shape")
parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")

args = parser.parse_args()
args.log_dir = logdir(args.log_dir)


PARAMS = {
    'epochs': args.epochs,
    'learning_rate': args.learning_rate,
    'batch_size': args.batch_size,
    'num_gpus' : args.num_gpus, 
    'num_classes' : args.num_classes, 
    'optimizer': args.optimizer,
    'loss_function': args.loss_function, 
    'scheduler': args.scheduler, 
    'model_architecture': args.model_architecture,  
    'data_path' : args.data_path,
    'pretrain_path' : args.pretrain_path,
    'excel_file' : args.excel_file,
    'log_dir' : args.log_dir,
}
PARAMS = get_model_parameters(PARAMS)

wandb.init(project="Multimodal-Bileductstone-Classifier", save_code=True, name = PARAMS['model_architecture'], config=PARAMS)


    
# Training and Valaidation
class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, device, data_path, excel_file, batch_size, log_dir):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.data_path = data_path
        self.excel_file = excel_file
        self.batch_size = batch_size
        self.log_dir = log_dir

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, Duct_diliatations_8mm, Duct_diliatation_10mm, Visible_stone_CT, Pancreatitis, targets in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            outputs = self.model(images.to(self.device), Duct_diliatations_8mm.to(self.device), Duct_diliatation_10mm.to(self.device), Visible_stone_CT.to(self.device), Pancreatitis.to(self.device))
            loss = self.loss_fn(outputs.squeeze(), targets.squeeze().float().to(self.device))  # Squeeze output and convert labels to float
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
            total_train += targets.size(0)
            correct_train += (predicted == targets.to(self.device)).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        return train_loss, train_acc

    def train(self, epochs, log_dir):
        train_loader, valid_loader = getloader_bc(self.data_path, self.excel_file, batch_size=self.batch_size, mode='train')
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        best_val_acc = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            if (epoch + 1) % 10 == 0:
                val_loss, val_acc = self.validate(valid_loader)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                wandb.log({"train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch+1)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

                # Save the best model based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print("Model Was Saved ! Current Best Accuracy: {} Current Accuracy: {}".format(best_val_acc, val_acc))
                    torch.save(self.model.module.state_dict(), f'{log_dir}/best_epoch_weights.pth')
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
            for images, Duct_diliatations_8mm, Duct_diliatation_10mm, Visible_stone_CT, Pancreatitis, targets in tqdm(valid_loader, desc="Validation"):
                outputs = self.model(images.to(self.device), Duct_diliatations_8mm.to(self.device), Duct_diliatation_10mm.to(self.device), Visible_stone_CT.to(self.device), Pancreatitis.to(self.device))
                loss = self.loss_fn(outputs.squeeze(), targets.squeeze().float().to(self.device))  # Squeeze output and convert labels to float
                val_running_loss += loss.item()
                predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
                total_val += targets.size(0)
                correct_val += (predicted == targets.to(self.device)).sum().item()

        val_loss = val_running_loss / len(valid_loader)
        val_acc = correct_val / total_val

        return val_loss, val_acc



# Create DataLoader and define model, optimizer, scheduler, loss_fn, and device
if args.data_shape == '3d':
    model = MultiModalbileductClassifier_3d(PARAMS['num_classes'], PARAMS['num_features'], PARAMS['model_architecture'], PARAMS['model_parameters'], PARAMS['pretrain_path'])
else:
    model = MultiModalbileductClassifier_2d(PARAMS['num_classes'], PARAMS['num_features'], PARAMS['model_architecture'], PARAMS['model_parameters'])
model = DataParallel(model, device_ids=[i for i in range(PARAMS['num_gpus'])]).to(device)

loss_fn, optimizer, scheduler = get_optimizer_loss_scheduler(PARAMS, model)

# Start time of training loop
start_training_time = timer()

trainer = Trainer(model, optimizer, scheduler, loss_fn, device, PARAMS['data_path'], PARAMS['excel_file'], PARAMS['batch_size'], PARAMS['log_dir'])
train_losses, val_losses, train_accs, val_accs = trainer.train(PARAMS['epochs'], PARAMS['log_dir'])

# End time of training loop
end_training_time = timer()

# Calculate total training time
total_training_time = (end_training_time - start_training_time) / 3600
print(f"Total Training Time: {total_training_time:.2f} hours")