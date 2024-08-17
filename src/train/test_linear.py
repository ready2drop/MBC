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
from src.utils.util import logdir, get_model_parameters, save_confusion_matrix_roc_curve


def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
seed_everything(1)


# Test
class Tester:
    def __init__(self, model, optimizer, scheduler, loss_fn, device, dict):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.epoch = dict['epochs']
        self.data_path = dict['data_path']
        self.excel_file = dict['excel_file']
        self.batch_size = dict['batch_size']
        self.log_dir = dict['log_dir']
        self.mode = dict['mode']
        self.modality = dict['modality']
        self.use_wandb = dict['use_wandb']
        
    def load_state_dict(self, weight_path, strict=False):
        sd = torch.load(weight_path, map_location="cpu")

        self.model.load_state_dict(sd, strict=strict)
        
        print(f"model parameters are loaded successed.")
        
    def test(self):
    
        test_loader = getloader_bc(self.data_path, self.excel_file, self.batch_size, self.mode, self.modality)
        self.model.eval()
        test_running_loss = 0.0
        correct_test = 0
        total_test = 0
        targets_all, predicted_all = [], []


        with torch.no_grad():
            with tqdm(total=len(test_loader), desc=f"Testing") as pbar: 
                if self.modality == 'mm':
                    for images, features, targets, _ in test_loader:
                        # outputs = self.model(images.to(self.device), [feature.to(self.device) for feature in features])
                        outputs = self.model(images.to(self.device), features.to(self.device))
                        loss = self.loss_fn(outputs.squeeze(), targets.squeeze().float().to(self.device))  # Squeeze output and convert labels to float
                        test_running_loss += loss.item()
                        predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
                        total_test += targets.size(0)
                        correct_test += (predicted == targets.to(self.device)).sum().item()
                        targets_all.append(targets.item())
                        predicted_all.append(predicted.item())
                        
                        if self.use_wandb:
                            wandb.log({"test_loss": loss.item(), "test_accuracy": (predicted == targets.to(self.device)).sum().item() / targets.size(0)})
                        pbar.update(1)
                        
                elif self.modality == 'image':
                    for images, targets, _ in test_loader:
                        outputs = self.model(images.to(self.device))
                        loss = self.loss_fn(outputs.squeeze(), targets.squeeze().float().to(self.device))  # Squeeze output and convert labels to float
                        test_running_loss += loss.item()
                        predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
                        total_test += targets.size(0)
                        correct_test += (predicted == targets.to(self.device)).sum().item()
                        targets_all.append(targets.item())
                        predicted_all.append(predicted.item())  
                                            
                        if self.use_wandb:
                            wandb.log({"test_loss": loss.item(), "test_accuracy": (predicted == targets.to(self.device)).sum().item() / targets.size(0)})
                        pbar.update(1)
                        
            test_loss = test_running_loss / len(test_loader)
            test_acc = correct_test / total_test

                
                
                    
        print(f"Test: Epoch Loss: {test_loss}, Epoch Accuracy: {test_acc}")
        save_confusion_matrix_roc_curve(targets_all, predicted_all, self.log_dir)     
        
        return test_loss, test_acc





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

parser = argparse.ArgumentParser(description="Multimodal Bile duct stone Classfier")
parser.add_argument("--epochs", default=10, type=int, help="Epoch")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--optimizer", default='adamw', type=str, help="Type of Optimizer") # 'adam', 'rmsprop'
parser.add_argument("--momentum", default=0.0, type=float, help="Add momentum for SGD optimizer")
parser.add_argument("--loss_function", default='BCE', type=str, help="Type of Loss function")
parser.add_argument("--scheduler", default='warmup_cosine', type=str, help="Type of Learning rate scheduler") # 'stepLR','CosineAnnealingLR'
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--num_gpus", default=8, type=int, help="Number of GPUs")
parser.add_argument("--num_classes", default=1, type=int, help="Assuming binary classification")
parser.add_argument("--use_parallel", action='store_true', help="Use Weights and Biases for logging")
parser.add_argument("--use_wandb", action='store_true', help="Use Weights and Biases for logging")
parser.add_argument("--model_architecture", default="efficientnet_b0", type=str, help="Model architecture")
parser.add_argument("--data_path", default='/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti/', type=str, help="Directory of dataset")
parser.add_argument("--pretrain_path", default='/home/irteam/rkdtjdals97-dcloud-dir/model_swinvit.pt', type=str, help="pretrained weight path")
parser.add_argument("--ckpt_path", default='/home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-05-21-15-53-train/best_epoch_weights.pth', type=str, help="finetuned weight path")
parser.add_argument("--excel_file", default='combined.csv', type=str, help="tabular data")
parser.add_argument("--data_shape", default='3d', type=str, help="Input data shape") # '3d','2d'
parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")
parser.add_argument("--mode", default='test', type=str, help="mode") # 'train', 'test'
parser.add_argument("--modality", default='mm', type=str, help="modality") # 'mm', 'image', 'tabular'

args = parser.parse_args()
args.log_dir = logdir(args.log_dir, args.mode, args.modality)

PARAMS = vars(args)
PARAMS = get_model_parameters(PARAMS)

if PARAMS['use_wandb'] == True:
    wandb.init(project="Multimodal-Bileductstone-Classifier-Test", save_code=True, name = f"{PARAMS['model_architecture']},{PARAMS['modality']}, {PARAMS['data_shape']}", config=PARAMS)

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
    
# Data parallel
if PARAMS['use_parallel']:
    model = DataParallel(model, device_ids=[i for i in range(PARAMS['num_gpus'])]).to(device)
else:
    model.to(device)    

# loss, optimizer, scheduler
loss_fn, optimizer, scheduler = get_optimizer_loss_scheduler(PARAMS, model)
tester = Tester(model, optimizer, scheduler, loss_fn, device, PARAMS)
tester.load_state_dict(PARAMS['ckpt_path'])
tester.test()



