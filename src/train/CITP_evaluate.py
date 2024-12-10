import torch
import torch.optim as optim
from torch.nn.parallel import DataParallel
import wandb
import argparse
import random
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.CITP_model import CITPModel, CITPModel_classifier
from src.dataset.bc_dataloader import getloader_bc
from src.utils.util import logdir, get_model_parameters, save_confusion_matrix_roc_curve
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix



def load_pretrained_encoders(dict, device):
    model = CITPModel(dict).to(device)
    model.image_encoder.load_state_dict(torch.load(f"{dict['pretrained_dir']}/CITP_Image_encoder_{dict['hidden_dim']}.pth"))
    model.tabular_encoder.load_state_dict(torch.load(f"{dict['pretrained_dir']}/CITP_Tabular_encoder_{dict['hidden_dim']}.pth"))
    return model.image_encoder, model.tabular_encoder

# Contrastive Image Tabular Pre-training(CITP)
def main(dict, device):
    data_loader = getloader_bc(dict['data_path'], dict['excel_file'], dict['batch_size'], dict['mode'], dict['modality'])
    # Load pretrained encoders
    image_encoder, tabular_encoder = load_pretrained_encoders(dict, device)

    # Initialize classifier
    model = CITPModel_classifier(image_encoder, tabular_encoder, dict['hidden_dim']).to(device)
    
    # Data parallel
    if dict['use_parallel']:
        model = DataParallel(model, device_ids=[int(gpu) for gpu in PARAMS['num_gpus'].split(",")] ).to(device)
    else:
        model.to(device) 
    
    model.eval()
    total_correct, total_samples = 0,0 
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for images, features, labels, _ in data_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            outputs = model(images, features)
            preds = (outputs > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            all_targets.append(labels.item())
            all_preds.append(preds.item())
            
            if dict.get('use_wandb', False):
                wandb.log({"test_accuracy": (preds == labels).sum().item() / labels.size(0)})
    
        
        
    # Calculate metrics
    accuracy = accuracy_score(all_targets, (np.array(all_preds) > 0.5).astype(int))
    auc = roc_auc_score(all_targets, all_preds)
    sensitivity = recall_score(all_targets, (np.array(all_preds) > 0.5).astype(int))  # Sensitivity (Recall)
    
    # Specificity calculation
    tn, fp, _, _ = confusion_matrix(all_targets, (np.array(all_preds) > 0.5).astype(int)).ravel()
    specificity = tn / (tn + fp)

    # Print metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    # Save results to Wandb if enabled
    if dict.get('use_wandb', False):
        wandb.log({
            "test_accuracy": accuracy,
            "test_auc": auc,
            "test_sensitivity": sensitivity,
            "test_specificity": specificity
        })              

    save_confusion_matrix_roc_curve(all_targets, all_preds, dict['log_dir'], dict['model_architecture'])
    
    
    
if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

    parser = argparse.ArgumentParser(description="Multimodal Bile duct stone Classfier")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--loss_function", default='BCE', type=str, help="Type of Loss function")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--num_gpus", default="0,1", type=str, help="Number of GPUs")
    parser.add_argument("--use_parallel", action='store_true', help="Use Weights and Biases for logging")
    parser.add_argument("--use_wandb", action='store_true', help="Use Weights and Biases for logging")
    parser.add_argument("--model_architecture", default="ViT", type=str, help="Model architecture")
    parser.add_argument("--data_path", default='/home/rkdtjdals97/datasets/DUMC_nifti_crop/', type=str, help="Directory of dataset")
    parser.add_argument("--excel_file", default='dumc_1024a.csv', type=str, help="tabular data")
    parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")
    parser.add_argument("--pretrained_dir", default='/home/rkdtjdals97/MBC/logs/2024-08-01-13-57-pretrain-mm', type=str, help="pretrained weight directory")
    parser.add_argument("--mode", default='eval', type=str, help="mode") # 'train', 'test'
    parser.add_argument("--modality", default='mm', type=str, help="modality") # 'mm', 'image', 'tabular'
    parser.add_argument("--hidden_dim", default=256, type=int, help="projection dimension") 
    parser.add_argument("--input_dim", default=19, type=int, help="tabular feature num") 

    args = parser.parse_args()
    args.log_dir = logdir(args.log_dir, args.mode, args.modality, args.model_architecture)

    PARAMS = vars(args)
    PARAMS = get_model_parameters(PARAMS)

    if PARAMS['use_wandb'] == True:
        wandb.init(project="Multimodal-Bileductstone-Classifier-CITP eval", save_code=True, name = f"{PARAMS['model_architecture']},{PARAMS['modality']}, {PARAMS['hidden_dim']}", config=PARAMS)

    main(PARAMS, device)