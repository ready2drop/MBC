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

from src.model.CITP_model import CITPModel
from src.dataset.bc_dataloader import getloader_bc
from src.utils.util import logdir, get_model_parameters, plot_tsne

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) 
    random.seed(seed) 
seed_everything(42)


# Contrastive Image Tabular Pre-training(CITP)
def main(dict, device):
    data_loader = getloader_bc(dict['data_path'], dict['excel_file'], dict['batch_size'], dict['mode'], dict['modality'])

    # Initialize models
    model = CITPModel(dict).to(device)

    # Data parallel
    if dict['use_parallel']:
        model = DataParallel(model, device_ids=[i for i in range(dict['num_gpus'])]).to(device)
    else:
        model.to(device) 
    
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize tracking variables
    best_loss = float('inf')  # Initialize to a very high value

    # Training loop
    for epoch in range(dict['epochs']):  # Adjust the number of epochs as needed
        train_loss, total_batches = 0.0, 0
        all_features, all_labels = [], []
        
        model.train()
        for images, features, labels, _ in data_loader:
            optimizer.zero_grad()
            image_features, tabular_features = model(images.to(device), features.to(device))
            # Contrastive loss
            loss = model.module.compute_contrastive_loss(image_features, tabular_features)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total_batches += 1
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            # Collect features for t-SNE
            combined_features = (image_features + tabular_features).detach().cpu().numpy()
            all_features.append(combined_features)
            all_labels.append(labels.detach().cpu().numpy())
            
            # Check if the current loss is the best (lowest) so far
            if loss.item() < best_loss:
                best_loss = loss.item()
                print("Model Saved ! Current Best loss: {} ".format(best_loss))
                torch.save(model.module.image_encoder.state_dict(), f"{dict['log_dir']}/CITP_Image_encoder_{dict['hidden_dim']}.pth")
                torch.save(model.module.tabular_encoder.state_dict(), f"{dict['log_dir']}/CITP_Tabular_encoder_{dict['hidden_dim']}.pth")
        
        # Calculate average loss for the epoch
        avg_train_loss = train_loss / total_batches
        if dict.get('use_wandb', False):
            wandb.log({"pretrain_loss": avg_train_loss}, step=epoch+1)
            
        # t-SNE visualization every 100 epochs
        if epoch % 100 == 0:
            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            plot_tsne(all_features, all_labels, epoch, dict['log_dir'])
    
    
    
if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

    parser = argparse.ArgumentParser(description="Multimodal Bile duct stone Classfier")
    parser.add_argument("--epochs", default=100, type=int, help="Epoch")
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
    parser.add_argument("--mode", default='pretrain', type=str, help="mode") # 'train', 'test'
    parser.add_argument("--modality", default='mm', type=str, help="modality") # 'mm', 'image', 'tabular'
    parser.add_argument("--hidden_dim", default=256, type=int, help="projection dimension") 
    parser.add_argument("--input_dim", default=19, type=int, help="tabular feature num") 

    args = parser.parse_args()
    args.log_dir = logdir(args.log_dir, args.mode, args.modality)

    PARAMS = vars(args)
    PARAMS = get_model_parameters(PARAMS)

    if PARAMS['use_wandb'] == True:
        wandb.init(project="Multimodal-Bileductstone-Classifier-CITP pretrain", save_code=True, name = f"{PARAMS['model_architecture']},{PARAMS['modality']}, {PARAMS['hidden_dim']}", config=PARAMS)

    main(PARAMS, device)