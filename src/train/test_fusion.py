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
from src.utils.util import logdir, get_model_parameters, save_confusion_matrix_roc_curve, plot_roc_and_calibration_test, save_feature_importance

# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from pytorch_tabnet.tab_model import TabNetClassifier
# from pytorch_tabnet.pretraining import TabNetPretrainer
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from sklearn.ensemble import *
# from sklearn.calibration import CalibratedClassifierCV
# from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score

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
parser.add_argument("--batch_size", default=50, type=int, help="Batch size")
parser.add_argument("--use_wandb", action='store_true', help="Use Weights and Biases for logging")
parser.add_argument("--model_architecture", default='ViT', type=str, help="Model architecture")
parser.add_argument("--data_path", default='/home/rkdtjdals97/datasets/DUMC_nifti_crop/', type=str, help="Directory of dataset")
parser.add_argument("--image_pretrain_path", default=None, type=str, help="pretrained weight path")
parser.add_argument("--ckpt_path", default='/home/rkdtjdals97/MBC/logs/2024-12-04-20-18-train-mm-ViT/', type=str, help="finetuned weight path")
parser.add_argument("--xgboost_ckpt_path", default='xgboost_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--lightgbm_ckpt_path", default='lightgbm_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--rf_ckpt_path", default='randomforest_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--adaboost_ckpt_path", default='adaboost_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--logreg_ckpt_path", default='logistic_regression_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--naivebayes_ckpt_path", default='naive_bayes_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--svm_ckpt_path", default='svm_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--decisiontree_ckpt_path", default='decision_tree_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--catboost_ckpt_path", default='catboost_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--tabnet_ckpt_path", default='tabnet_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--stacking_ckpt_path", default='stacking_model_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--calibration_stacking_ckpt_path", default='calibrated_stacking_model_model_12.pkl', type=str, help="finetuned weight path")
parser.add_argument("--excel_file", default='dumc_1024a.csv', type=str, help="tabular data")
parser.add_argument("--data_shape", default='3d', type=str, help="Input data shape") # '3d','2d'
parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")
parser.add_argument("--mode", default='test', type=str, help="mode") # 'train', 'test'
parser.add_argument("--modality", default='mm', type=str, help="modality") # 'mm', 'image', 'tabular'
parser.add_argument("--output_dim", default=128, type=int, help="output dimension") # output dimension of each encoder
parser.add_argument("--input_dim", default=12, type=int, help="num_features") # tabular features
parser.add_argument("--fusion", default='early', type=str, help="num_features") # 'early','intermediate', 'late'
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


test_loader = getloader_bc(PARAMS['data_path'], PARAMS['excel_file'], PARAMS['batch_size'], PARAMS['mode'], PARAMS['modality'], PARAMS['phase'])

if PARAMS['fusion'] == 'late':
    image_encoder.eval()
    tabular_encoder.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for images, features, targets, _ in tqdm(test_loader, desc="TestData_Generation"):
            images, features, targets = images.to(device), features.to(device), targets.to(device)
            
            image_predicts = image_encoder(images)
            tabular_predicts = tabular_encoder(features)

            # Apply sigmoid to get probabilities
            image_probs = torch.sigmoid(image_predicts).squeeze()
            tabular_probs = torch.sigmoid(tabular_predicts).squeeze()

            # Aggregate predictions (average in this example)
            combined_probs = (image_probs + tabular_probs) / 2
            
            test_preds.append(combined_probs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
            
    # Convert lists to numpy arrays
    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    # Convert probabilities to binary predictions
    test_preds_binary = (test_preds > 0.5).astype(int)

    # Evaluate metrics
    accuracy = accuracy_score(test_targets, test_preds_binary)
    auc = roc_auc_score(test_targets, test_preds)
    sensitivity = recall_score(test_targets, test_preds_binary)  # Sensitivity (Recall)

    # Specificity calculation
    tn, fp, _, _ = confusion_matrix(test_targets, test_preds_binary).ravel()
    specificity = tn / (tn + fp)

    # Print results
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'Sensitivity (Recall): {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')

else:
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
    # Get feature names from tabular data (X_test should include tabular features)
    feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
    
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
        # # Save feature importance if applicable (for XGBoost, LightGBM, RandomForest)
        # if hasattr(model, 'feature_importances_'):
        #     save_feature_importance(model.feature_importances_, feature_names, model_name, args.log_dir)
        
        # return test_auc

    model_paths = {
    'XGBoost': os.path.join(PARAMS['ckpt_path'], PARAMS['xgboost_ckpt_path']),
    'LightGBM': os.path.join(PARAMS['ckpt_path'], PARAMS['lightgbm_ckpt_path']),
    'RandomForest': os.path.join(PARAMS['ckpt_path'], PARAMS['rf_ckpt_path']),
    'AdaBoost': os.path.join(PARAMS['ckpt_path'], PARAMS['adaboost_ckpt_path']),
    'Logistic Regression': os.path.join(PARAMS['ckpt_path'], PARAMS['logreg_ckpt_path']),
    'Naive Bayes': os.path.join(PARAMS['ckpt_path'], PARAMS['naivebayes_ckpt_path']),
    'SVM': os.path.join(PARAMS['ckpt_path'], PARAMS['svm_ckpt_path']),
    'Decision Tree': os.path.join(PARAMS['ckpt_path'], PARAMS['decisiontree_ckpt_path']),
    'CatBoost': os.path.join(PARAMS['ckpt_path'], PARAMS['catboost_ckpt_path']),
    'TabNet': os.path.join(PARAMS['ckpt_path'], PARAMS['tabnet_ckpt_path']),
    'Stacking Model': os.path.join(PARAMS['ckpt_path'], PARAMS['stacking_ckpt_path']),
    'Calibrated Stacking Model': os.path.join(PARAMS['ckpt_path'], PARAMS['calibration_stacking_ckpt_path'])
    }

    # Function to load and evaluate models
    def load_and_evaluate_models(model_paths, X_test, y_test):
        models = []
        model_names = []
        
        for model_name, ckpt_path in model_paths.items():
            # Load the model
            with open(ckpt_path, 'rb') as f:
                model = pickle.load(f)
            
            # Append the model and its name to the lists
            models.append(model)
            model_names.append(model_name)
            
            # Evaluate the model
            print(f"Evaluating {model_name} Model")
            evaluate_model(model, X_test, y_test, model_name)
        
        return models, model_names
            
    # Call the function to load and evaluate models
    models, model_names = load_and_evaluate_models(model_paths, X_test, y_test)
    
    

    # Call the function to plot ROC and calibration curves for all models
    plot_roc_and_calibration_test(models, X_test, y_test, args.log_dir, model_names)




