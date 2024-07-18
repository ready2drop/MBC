import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from dataset.bc_dataloader import getloader_bc
from model.image_encoder import ImageEncoder3D_earlyfusion, ImageEncoder3D_latefusion
from model.tabular_encoder import TabularEncoder_earlyfusion, TabularEncoder_latefusion
from utils.loss import get_optimizer_loss_scheduler
from utils.util import logdir, get_model_parameters

import pickle  # For saving models
from tqdm import tqdm
import numpy as np
import random
import wandb
import argparse
import warnings
warnings.filterwarnings('ignore')

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

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
parser.add_argument("--data_path", default='/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part4_nifti_crop_ver2/', type=str, help="Directory of dataset")
parser.add_argument("--image_pretrain_path", default='/home/irteam/rkdtjdals97-dcloud-dir/MBC/pretrain/model_swinvit.pt', type=str, help="Image pretrained weight path")
parser.add_argument("--tabnet_pretrain_path", default='/home/irteam/rkdtjdals97-dcloud-dir/MBC/pretrain/pretrain_model.zip', type=str, help="TabNet pretrained weight path")
parser.add_argument("--excel_file", default='dumc_0702.csv', type=str, help="tabular data")
parser.add_argument("--data_shape", default='3d', type=str, help="Input data shape") # '3d','2d'
parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")
parser.add_argument("--mode", default='train', type=str, help="mode") # 'train', 'test'
parser.add_argument("--modality", default='mm', type=str, help="modality") # 'mm', 'image', 'tabular'
parser.add_argument("--output_dim", default=1, type=int, help="output dimension") # output dimension of each encoder
parser.add_argument("--input_dim", default=17, type=int, help="num_features") # tabular features
parser.add_argument("--fusion", default='late', type=str, help="num_features") # 'early','intermediate', 'late'

args = parser.parse_args()
args.log_dir = logdir(args.log_dir, args.mode, args.modality)

PARAMS = vars(args)
PARAMS = get_model_parameters(PARAMS)

if PARAMS['use_wandb'] == True:
    wandb.init(project="Multimodal-Bileductstone-Classifier", save_code=True, name = f"{PARAMS['model_architecture']},{PARAMS['modality']}, {PARAMS['data_shape']}", config=PARAMS)


if PARAMS['fusion'] == 'early':
    # Image Encoder
    image_encoder = ImageEncoder3D_earlyfusion(PARAMS).to(device)
    # Tabular Encoder
    tabular_encoder = TabularEncoder_earlyfusion(PARAMS).to(device)
    # Combined dimension
    combined_dim = PARAMS['input_dim']*2
    
elif PARAMS['fusion'] == 'intermediate': 
    # Image Encoder
    image_encoder = ImageEncoder3D_latefusion(PARAMS).to(device)
    # Tabular Encoder
    tabular_encoder = TabularEncoder_earlyfusion(PARAMS).to(device)
    # Combined dimension
    combined_dim = PARAMS['input_dim'] + PARAMS['output_dim']
else: 
    # Image Encoder
    image_encoder = ImageEncoder3D_latefusion(PARAMS).to(device)
    # Tabular Encoder
    tabular_encoder = TabularEncoder_latefusion(PARAMS).to(device)
    


train_loader, valid_loader = getloader_bc(PARAMS['data_path'], PARAMS['excel_file'], PARAMS['batch_size'], PARAMS['mode'], PARAMS['modality'])


if PARAMS['fusion'] == 'late':
    image_encoder.eval()
    tabular_encoder.eval()
    
    train_preds, val_preds, train_targets, val_targets = [], [], [], []

    with torch.no_grad():
        for images, features, targets, _ in tqdm(train_loader, desc="TrainData_Generation"):
            images, features, targets = images.to(device), features.to(device), targets.to(device)
            
            image_predicts = image_encoder(images)
            tabular_predicts = tabular_encoder(features)

            # Apply sigmoid to get probabilities
            image_probs = torch.sigmoid(image_predicts).squeeze()
            tabular_probs = torch.sigmoid(tabular_predicts).squeeze()

            # Aggregate predictions (average in this example)
            combined_probs = (image_probs + tabular_probs) / 2
            
            train_preds.append(combined_probs.cpu().numpy())
            train_targets.append(targets.cpu().numpy())

    with torch.no_grad():
        for images, features, targets, _ in tqdm(valid_loader, desc="ValidData_Generation"):
            images, features, targets = images.to(device), features.to(device), targets.to(device)
            
            image_predicts = image_encoder(images)
            tabular_predicts = tabular_encoder(features)

            # Apply sigmoid to get probabilities
            image_probs = torch.sigmoid(image_predicts).squeeze()
            tabular_probs = torch.sigmoid(tabular_predicts).squeeze()

            # Aggregate predictions (average in this example)
            combined_probs = (image_probs + tabular_probs) / 2
            
            val_preds.append(combined_probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    # Convert lists to numpy arrays
    train_preds = np.concatenate(train_preds, axis=0)
    train_targets = np.concatenate(train_targets, axis=0)
    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Convert probabilities to binary predictions
    train_preds_binary = (train_preds > 0.5).astype(int)
    val_preds_binary = (val_preds > 0.5).astype(int)

    # Evaluate
    train_accuracy = accuracy_score(train_targets, train_preds_binary)
    val_accuracy = accuracy_score(val_targets, val_preds_binary)

    print(f'Train Accuracy: {train_accuracy}')
    print(f'Validation Accuracy: {val_accuracy}')
    
else:
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

    # Train and evaluate models
    def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
        model.fit(X_train, y_train)
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, y_train_pred)
        val_auc = roc_auc_score(y_val, y_val_pred)
        
        train_acc = accuracy_score(y_train, (y_train_pred > 0.5).astype(int))
        val_acc = accuracy_score(y_val, (y_val_pred > 0.5).astype(int))
        
        train_logloss = log_loss(y_train, y_train_pred)
        val_logloss = log_loss(y_val, y_val_pred)
        
        print(f'{model_name} - Train AUC: {train_auc}, Val AUC: {val_auc}')
        print(f'{model_name} - Train Accuracy: {train_acc}, Val Accuracy: {val_acc}')
        print(f'{model_name} - Train LogLoss: {train_logloss}, Val LogLoss: {val_logloss}')
        
    # XGBoost
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    train_and_evaluate_model(xgb_model, X_train, y_train, X_val, y_val, 'XGBoost')
    with open(f"{PARAMS['log_dir']}/xgb_model.pkl", 'wb') as f:
        pickle.dump(xgb_model,  f)  # Save the XGBoost model

    # LightGBM
    lgbm_model = LGBMClassifier(random_state=42,verbose_eval = -1)
    train_and_evaluate_model(lgbm_model, X_train, y_train, X_val, y_val, 'LightGBM')
    with open(f"{PARAMS['log_dir']}/lgbm_model.pkl", 'wb') as f:
        pickle.dump(lgbm_model,  f)  # Save the XGBoost model

    # RandomForest
    rf_model = RandomForestClassifier(random_state=42)
    train_and_evaluate_model(rf_model, X_train, y_train, X_val, y_val, 'RandomForest')
    with open(f"{PARAMS['log_dir']}/rf_model.pkl", 'wb') as f:
        pickle.dump(rf_model,  f)  # Save the XGBoost model

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

    # Load pretrained model
    # loaded_pretrain = TabNetPretrainer()
    # loaded_pretrain.load_model(PARAMS['tabnet_pretrain_path'])
    # print('All weights are loaded!')    

    tabnet_clf = TabNetClassifier(**tabnet_params
                        )
            
    tabnet_clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train),(X_val, y_val)],
        eval_name=['train', 'valid'],
        eval_metric=['auc', 'accuracy', 'balanced_accuracy', 'logloss'],
        max_epochs=1000,
        patience=50,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        # from_unsupervised=loaded_pretrain
    ) 

    # save tabnet model
    saving_path_name = f"{PARAMS['log_dir']}/tabnet_{PARAMS['input_dim']}"
    saved_filepath = tabnet_clf.save_model(saving_path_name)


