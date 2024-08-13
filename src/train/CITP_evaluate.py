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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) 
    random.seed(seed) 
seed_everything(42)

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
        model = DataParallel(model, device_ids=[i for i in range(dict['num_gpus'])]).to(device)
    else:
        model.to(device) 
    
    model.eval()
    total_correct, total_samples = 0,0 
    targets_all, predicted_all = [], []
    
    with torch.no_grad():
        for images, features, labels, _ in data_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            outputs = model(images, features)
            preds = (outputs > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            targets_all.append(labels.item())
            predicted_all.append(preds.item())
            
            if dict.get('use_wandb', False):
                wandb.log({"test_accuracy": (preds == labels).sum().item() / labels.size(0)})
    
        test_acc = total_correct / total_samples             
                  
    print(f"Test: Epoch Accuracy: {test_acc}")
    save_confusion_matrix_roc_curve(targets_all, predicted_all, dict['log_dir'], dict['model_architecture'])
    
    
    
if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

    parser = argparse.ArgumentParser(description="Multimodal Bile duct stone Classfier")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--loss_function", default='BCE', type=str, help="Type of Loss function")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--num_gpus", default=8, type=int, help="Number of GPUs")
    parser.add_argument("--use_parallel", action='store_true', help="Use Weights and Biases for logging")
    parser.add_argument("--use_wandb", action='store_true', help="Use Weights and Biases for logging")
    parser.add_argument("--model_architecture", default="ViT", type=str, help="Model architecture")
    parser.add_argument("--data_path", default='/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part5_nifti_crop/', type=str, help="Directory of dataset")
    parser.add_argument("--excel_file", default='dumc_0730a.csv', type=str, help="tabular data")
    parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")
    parser.add_argument("--pretrained_dir", default='/home/irteam/rkdtjdals97-dcloud-dir/MBC/logs/2024-08-01-13-57-pretrain-mm', type=str, help="pretrained weight directory")
    parser.add_argument("--mode", default='eval', type=str, help="mode") # 'train', 'test'
    parser.add_argument("--modality", default='mm', type=str, help="modality") # 'mm', 'image', 'tabular'
    parser.add_argument("--hidden_dim", default=256, type=int, help="projection dimension") 
    parser.add_argument("--input_dim", default=19, type=int, help="tabular feature num") 

    args = parser.parse_args()
    args.log_dir = logdir(args.log_dir, args.mode, args.modality)

    PARAMS = vars(args)
    PARAMS = get_model_parameters(PARAMS)

    if PARAMS['use_wandb'] == True:
        wandb.init(project="Multimodal-Bileductstone-Classifier-CITP eval", save_code=True, name = f"{PARAMS['model_architecture']},{PARAMS['modality']}, {PARAMS['hidden_dim']}", config=PARAMS)

    main(PARAMS, device)