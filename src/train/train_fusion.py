import sys
import os
from tqdm import tqdm
import numpy as np
import random
import wandb
import argparse
import pickle  # For saving models
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch

from src.dataset.bc_dataloader import getloader_bc
from src.model.image_encoder import ImageEncoder3D_earlyfusion, ImageEncoder3D_latefusion
from src.model.tabular_encoder import TabularEncoder_earlyfusion, TabularEncoder_latefusion
from src.utils.util import logdir, get_model_parameters

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import *
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV


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
parser.add_argument("--batch_size", default=100, type=int, help="Batch size")
parser.add_argument("--use_wandb", action='store_true', help="Use Weights and Biases for logging")
parser.add_argument("--model_architecture", default='ViT', type=str, help="Model architecture")
parser.add_argument("--data_path", default='/home/rkdtjdals97/datasets/DUMC_nifti_crop/', type=str, help="Directory of dataset")
parser.add_argument("--image_pretrain_path", default=None, type=str, help="Image pretrained weight path")
parser.add_argument("--tabnet_pretrain_path", default='/home/rkdtjdals97/MBC/pretrain/pretrain_model.zip', type=str, help="TabNet pretrained weight path")
parser.add_argument("--excel_file", default='dumc_1024a.csv', type=str, help="tabular data")
parser.add_argument("--data_shape", default='3d', type=str, help="Input data shape") # '3d','2d'
parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")
parser.add_argument("--mode", default='train', type=str, help="mode") # 'train', 'test'
parser.add_argument("--modality", default='mm', type=str, help="modality") # 'mm', 'image', 'tabular'
parser.add_argument("--output_dim", default=128, type=int, help="output dimension") # output dimension of each encoder
parser.add_argument("--input_dim", default=12, type=int, help="num_features") # all tabular features minus 2(image_path, target) 
parser.add_argument("--fusion", default='early', type=str, help="fusion method") # 'early','intermediate', 'late'
parser.add_argument("--phase", default='combine', type=str, help="CT phase") # 'portal', 'pre-enhance', 'combine'

args = parser.parse_args()
args.log_dir = logdir(args.log_dir, args.mode, args.modality, args.model_architecture)

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
    


train_loader, valid_loader = getloader_bc(PARAMS['data_path'], PARAMS['excel_file'], PARAMS['batch_size'], PARAMS['mode'], PARAMS['modality'], PARAMS['phase'])


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
    
    # Define all models in a dictionary
    models = {
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "RandomForest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(random_state=42, probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "TabNet": TabNetClassifier(verbose=0)
    }
    # Create a Stacking Ensemble Model
    stacking_model = StackingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        final_estimator=LogisticRegression()
    )

    # Train the stacking model
    stacking_model.fit(X_train, y_train)

    # Add calibrated stacking model to the list of models
    models["Stacking Model"] = stacking_model
    
    # Calibrate the Stacking Model
    calibrated_stacking_model = CalibratedClassifierCV(stacking_model, cv="prefit", method="sigmoid")
    calibrated_stacking_model.fit(X_train, y_train)

    # Add calibrated stacking model to the list of models
    models["Calibrated Stacking Model"] = calibrated_stacking_model
    
    # Function to train, evaluate, and save each model
    def process_model(model_name, model, X_train, y_train, X_val, y_val, log_dir, input_dim):
        train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name)
        with open(f"{log_dir}/{model_name.lower().replace(' ', '_')}_model_{input_dim}.pkl", 'wb') as f:
            pickle.dump(model, f)

    # Iterate through models and process them
    for model_name, model in models.items():
        process_model(model_name, model, X_train, y_train, X_val, y_val, PARAMS['log_dir'], PARAMS['input_dim'])


