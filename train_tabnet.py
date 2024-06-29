import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from dataset.bc_dataloader import getloader_bc
from model.image_encoder import ImageEncoder3D
from model.tabular_encoder import TabularEncoder
from pytorch_tabnet.tab_model import TabNetClassifier

from model.tabnet import TabNet
from utils.loss import get_optimizer_loss_scheduler
from utils.util import logdir, get_model_parameters

from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
import random
import wandb
import argparse
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) 
    random.seed(seed) 
seed_everything(42)
    



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

parser = argparse.ArgumentParser(description="Multimodal Bile duct stone Classfier")
parser.add_argument("--epochs", default=100, type=int, help="Epoch")
parser.add_argument("--val_every", default=10, type=int, help="Learning rate")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--optimizer", default='adamw', type=str, help="Type of Optimizer") # 'adam', 'rmsprop'
parser.add_argument("--momentum", default=0.0, type=float, help="Add momentum for SGD optimizer")
parser.add_argument("--loss_function", default='BCE', type=str, help="Type of Loss function")
parser.add_argument("--scheduler", default='warmup_cosine', type=str, help="Type of Learning rate scheduler") # 'stepLR','CosineAnnealingLR'
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--num_gpus", default=8, type=int, help="Number of GPUs")
parser.add_argument("--num_classes", default=1, type=int, help="Assuming binary classification")
parser.add_argument("--use_parallel", action='store_true', help="Use Weights and Biases for logging")
parser.add_argument("--use_wandb", action='store_true', help="Use Weights and Biases for logging")
parser.add_argument("--model_architecture", default='SwinUNETR', type=str, help="Model architecture")
parser.add_argument("--data_path", default='/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part3_nifti_crop/', type=str, help="Directory of dataset")
parser.add_argument("--pretrain_path", default='/home/irteam/rkdtjdals97-dcloud-dir/model_swinvit.pt', type=str, help="pretrained weight path")
parser.add_argument("--excel_file", default='dumc_0618.csv', type=str, help="tabular data")
parser.add_argument("--data_shape", default='3d', type=str, help="Input data shape") # '3d','2d'
parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")
parser.add_argument("--mode", default='train', type=str, help="mode") # 'train', 'test'
parser.add_argument("--modality", default='mm', type=str, help="modality") # 'mm', 'image', 'tabular'
parser.add_argument("--output_dim", default=128, type=int, help="output dimension") # output dimension of each encoder
parser.add_argument("--input_dim", default=18, type=int, help="num_features") # tabular features

args = parser.parse_args()
args.log_dir = logdir(args.log_dir, args.mode, args.modality)

PARAMS = vars(args)
PARAMS = get_model_parameters(PARAMS)

if PARAMS['use_wandb'] == True:
    wandb.init(project="Multimodal-Bileductstone-Classifier", save_code=True, name = f"{PARAMS['model_architecture']},{PARAMS['modality']}, {PARAMS['data_shape']}", config=PARAMS)


# Image Encoder
image_encoder = ImageEncoder3D(PARAMS).to(device)
# Tabular Encoder
tabular_encoder = TabularEncoder(PARAMS).to(device)

# Combined dimension
combined_dim = PARAMS['output_dim']*2

train_loader, valid_loader = getloader_bc(PARAMS['data_path'], PARAMS['excel_file'], PARAMS['batch_size'], PARAMS['mode'], PARAMS['modality'])

image_encoder.eval()
tabular_encoder.eval()

X_train, X_val, y_train, y_val = [], [], [], []

with torch.no_grad():
    for images, features, targets, _ in tqdm(train_loader, desc="TrainData_Generation"):
        images, features = images.to(device), features.to(device)
        

        image_features = image_encoder(images)
        tabular_features = tabular_encoder(features)

        combined = torch.cat((image_features, tabular_features), dim=1)
        X_train.append(combined.cpu().numpy())
        y_train.append(targets.cpu().numpy())

with torch.no_grad():
    for images, features, targets, _ in tqdm(valid_loader, desc="ValidData_Generation"):
        images, features = images.to(device), features.to(device)
        

        image_features = image_encoder(images)
        tabular_features = tabular_encoder(features)

        combined = torch.cat((image_features, tabular_features), dim=1)
        X_val.append(combined.cpu().numpy())
        y_val.append(targets.cpu().numpy())

# Convert combined features and targets to numpy arrays
X_train = np.vstack(X_train)
y_train = np.hstack(y_train)
X_val = np.vstack(X_val)
y_val = np.hstack(y_val)

print('X_train shape : ', X_train.shape, 'y_train shape : ', y_train.shape, 'X_test shape : ', X_val.shape, 'y_test shape : ', y_val.shape) 

# Set tabnet_params
tabnet_params = {"cat_emb_dim":2,
            "optimizer_fn":torch.optim.Adam,
            "optimizer_params":dict(lr=2e-2),
            "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                            "gamma":0.9},
            "scheduler_fn":torch.optim.lr_scheduler.StepLR,
            "mask_type":'sparsemax', # "entmax"
        }

# Adjust tabnet_params to include combined_dim
tabnet_params['input_dim'] = combined_dim
tabnet_params['output_dim'] = PARAMS['num_classes']
     
tabnet_clf = TabNetClassifier(**tabnet_params
                      )
        
tabnet_clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train),(X_val, y_val)],
    eval_name=['train', 'valid'],
    eval_metric=['auc', 'accuracy', 'balanced_accuracy', 'logloss'],
    max_epochs=1000,
    patience=100,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False
) 

# save tabnet model
saving_path_name = f'{args.log_dir}/best_epoch_weights'
saved_filepath = tabnet_clf.save_model(saving_path_name)


