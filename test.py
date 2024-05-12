import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from dataset.dataloader import getloader
from dataset.bc_dataloader import getloader_bc
from model.mbc import MultiModalbileductClassifier_2d, MultiModalbileductClassifier_3d
from utils.loss import get_optimizer_loss_scheduler
from utils.util import logdir, get_model_parameters, save_confusion_matrix_roc_curve

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
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_gpus", default=8, type=int, help="Number of GPUs")
parser.add_argument("--num_classes", default=1, type=int, help="Assuming binary classification")
parser.add_argument("--optimizer", default='adam', type=str, help="Type of Optimizer") # 'adam', 'rmsprop'
parser.add_argument("--loss_function", default='BCE', type=str, help="Type of Loss function")
parser.add_argument("--scheduler", default='StepLR', type=str, help="Type of Learning rate scheduler") # 'stepLR','CosineAnnealingLR'
parser.add_argument("--momentum", default=0.0, type=float, help="Add momentum for SGD optimizer")
parser.add_argument("--model_architecture", default="efficientnet_b0", type=str, help="Model architecture")
parser.add_argument("--data_path", default='/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti/', type=str, help="Directory of data")
parser.add_argument("--pretrain_path", default='/home/irteam/rkdtjdals97-dcloud-dir/model_swinvit.pt', type=str, help="pretrained weight path")
parser.add_argument("--ckpt_path", default='/home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-05-11-00-10/best_epoch_weights.pth', type=str, help="pretrained weight path")
parser.add_argument("--excel_file", default='bileduct_data_20240508b.xlsx', type=str, help="excel data")
parser.add_argument("--data_shape", default='3d', type=str, help="Input data shape")
parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")
parser.add_argument("--mode", default='test', type=str, help="mode")

args = parser.parse_args()
args.log_dir = logdir(args.log_dir, args.mode)

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
    'ckpt_path' : args.ckpt_path,
    'excel_file' : args.excel_file,
    'log_dir' : args.log_dir,
    'mode' : args.mode,
}
PARAMS = get_model_parameters(PARAMS)

wandb.init(project="Multimodal-Bileductstone-Classifier-Test", save_code=True, name = PARAMS['model_architecture'], config=PARAMS)


    
# Training and Valaidation
class Tester:
    def __init__(self, model, optimizer, scheduler, loss_fn, device, data_path, excel_file, batch_size, log_dir, mode):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.data_path = data_path
        self.excel_file = excel_file
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.mode = mode   
        
    def load_state_dict(self, weight_path, strict=False):
        sd = torch.load(weight_path, map_location="cpu")

        self.model.load_state_dict(sd, strict=strict)
        
        print(f"model parameters are loaded successed.")
        
    def test(self):
        self.model.eval()
        test_running_loss = 0.0
        correct_test = 0
        total_test = 0
        targets_all, predicted_all = [], []

        test_loader = getloader_bc(self.data_path, self.excel_file, batch_size=self.batch_size, mode=self.mode)
        
        with torch.no_grad():
            for images, Duct_diliatations_8mm, Duct_diliatation_10mm, Visible_stone_CT, Pancreatitis, targets in tqdm(test_loader, desc="Test"):
                outputs = self.model(images.to(self.device), Duct_diliatations_8mm.to(self.device), Duct_diliatation_10mm.to(self.device), Visible_stone_CT.to(self.device), Pancreatitis.to(self.device))
                loss = self.loss_fn(outputs.squeeze(), targets.squeeze().float().to(self.device))  # Squeeze output and convert labels to float
                test_running_loss += loss.item()
                predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
                total_test += targets.size(0)
                correct_test += (predicted == targets.to(self.device)).sum().item()
                targets_all.append(targets.item())
                predicted_all.append(predicted.item())
                wandb.log({"test_loss": loss.item(), "test_accuracy": (predicted == targets.to(self.device)).sum().item() / targets.size(0)})

        test_loss = test_running_loss / len(test_loader)
        test_acc = correct_test / total_test
        save_confusion_matrix_roc_curve(targets_all, predicted_all, self.log_dir)
        print("Test : Accuracy: {} Loss: {}".format(test_acc, test_loss))
        
        
        return test_loss, test_acc, 





# Create DataLoader and define model, optimizer, scheduler, loss_fn, and device
if args.data_shape == '3d':
    model = MultiModalbileductClassifier_3d(PARAMS['num_classes'], PARAMS['num_features'], PARAMS['model_architecture'], PARAMS['model_parameters'], PARAMS['pretrain_path']).to(device)
else:
    model = MultiModalbileductClassifier_2d(PARAMS['num_classes'], PARAMS['num_features'], PARAMS['model_architecture'], PARAMS['model_parameters']).to(device)
# model = DataParallel(model, device_ids=[i for i in range(PARAMS['num_gpus'])]).to(device)

loss_fn, optimizer, scheduler = get_optimizer_loss_scheduler(PARAMS, model)

# Start time of training loop
start_training_time = timer()

tester = Tester(model, optimizer, scheduler, loss_fn, device, PARAMS['data_path'], PARAMS['excel_file'], PARAMS['batch_size'], PARAMS['log_dir'], PARAMS['mode'])
tester.load_state_dict(PARAMS['ckpt_path'])
test_loss, test_acc = tester.test()

# End time of training loop
end_training_time = timer()

# Calculate total training time
total_training_time = (end_training_time - start_training_time) / 3600
print(f"Total Training Time: {total_training_time:.2f} hours")