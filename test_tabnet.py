import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from dataset.bc_dataloader import getloader_bc
from model.image_encoder import ImageEncoder3D
from model.tabular_encoder import TabularEncoder
from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score

from utils.loss import get_optimizer_loss_scheduler
from utils.util import logdir, get_model_parameters, save_confusion_matrix_roc_curve

from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
import random
import wandb
import argparse
import pickle
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
parser.add_argument("--data_path", default='/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part4_nifti_crop_ver2/', type=str, help="Directory of dataset")
parser.add_argument("--image_pretrain_path", default='/home/irteam/rkdtjdals97-dcloud-dir/MBC/pretrain/model_swinvit.pt', type=str, help="pretrained weight path")
parser.add_argument("--tabnet_ckpt_path", default='/home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-07-09-20-04-train-mm/tabnet_128.zip', type=str, help="finetuned weight path")
parser.add_argument("--xgboost_ckpt_path", default='/home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-07-09-20-04-train-mm/xgb_model.pkl', type=str, help="finetuned weight path")
parser.add_argument("--lightgbm_ckpt_path", default='/home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-07-09-20-04-train-mm/lgbm_model.pkl', type=str, help="finetuned weight path")
parser.add_argument("--rf_ckpt_path", default='/home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-07-09-20-04-train-mm/rf_model.pkl', type=str, help="finetuned weight path")
parser.add_argument("--excel_file", default='dumc_0702.csv', type=str, help="tabular data")
parser.add_argument("--data_shape", default='3d', type=str, help="Input data shape") # '3d','2d'
parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")
parser.add_argument("--mode", default='test', type=str, help="mode") # 'train', 'test'
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

test_loader = getloader_bc(PARAMS['data_path'], PARAMS['excel_file'], PARAMS['batch_size'], PARAMS['mode'], PARAMS['modality'])

image_encoder.eval()
tabular_encoder.eval()

X_test, y_test = [], []

with torch.no_grad():
    for images, features, targets, _ in tqdm(test_loader, desc="TestData_Generation"):
        images, features = images.to(device), features.to(device)
        

        image_features = image_encoder(images)
        tabular_features = tabular_encoder(features)

        combined = torch.cat((image_features, tabular_features), dim=1)
        X_test.append(combined.cpu().numpy())
        y_test.append(targets.cpu().numpy())

# Convert combined features and targets to numpy arrays
X_test = np.vstack(X_test)
y_test = np.hstack(y_test)

print('X_test shape : ', X_test.shape, 'y_test shape : ', y_test.shape) 

def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, preds)

    # Convert probabilities to binary predictions
    test_preds_binary = (preds >= 0.5).astype(int)

    save_confusion_matrix_roc_curve(y_test, test_preds_binary, args.log_dir, model_name) 
    # Confusion matrix for the test set
    conf_matrix_test = confusion_matrix(y_test, test_preds_binary)
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, test_preds_binary)
    roc_auc = auc(fpr, tpr)
    # Metrics for the test set
    accuracy_test = accuracy_score(y_test, test_preds_binary)
    sensitivity_test = recall_score(y_test, test_preds_binary)  # Sensitivity is also called recall
    specificity_test = recall_score(y_test, test_preds_binary, pos_label=0)
    precision_test = precision_score(y_test, test_preds_binary)

    print("\nConfusion Matrix for Test Set:")
    print(conf_matrix_test)
    print('ROC curve (area = %0.2f)' % roc_auc)
    print(f"Test Accuracy: {accuracy_test}")
    print(f"Test Sensitivity (Recall): {sensitivity_test}")
    print(f"Test Specificity: {specificity_test}")
    print(f"Test Precision: {precision_test}")
    print(f"FINAL TEST SCORE FOR MBC : {test_auc}")
    return test_auc

# Load and evaluate RandomForest model
with open(PARAMS['xgboost_ckpt_path'], 'rb') as f:
    xgb_model = pickle.load(f)
print("Evaluating XGBoost Model")
evaluate_model(xgb_model, X_test, y_test, 'XGBoost')

# Load and evaluate RandomForest model
with open(PARAMS['lightgbm_ckpt_path'], 'rb') as f:
    lgbm_model = pickle.load(f)
print("Evaluating LightGBM Model")
evaluate_model(lgbm_model, X_test, y_test, 'LightGBM')

# Load and evaluate RandomForest model
with open(PARAMS['rf_ckpt_path'], 'rb') as f:
    rf_model = pickle.load(f)
print("Evaluating RandomForest Model")
evaluate_model(rf_model, X_test, y_test, 'RandomForest')

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

tabnet_clf.load_model(PARAMS['tabnet_ckpt_path'])        
print("Evaluating TabNet Model")
evaluate_model(tabnet_clf, X_test, y_test, 'TabNet')


