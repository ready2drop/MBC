import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import argparse
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from MBC.src.model.CITP import ContrastiveImageEncoder, ContrastiveTabularEncoder
from src.dataset.bc_dataloader import getloader_bc
from src.utils.util import logdir, get_model_parameters

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        features1 = nn.functional.normalize(features1, dim=-1)
        features2 = nn.functional.normalize(features2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # Generate labels: for each sample, the positive sample index is the same as the row index
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=features1.device)
        
        # Compute the contrastive loss
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        
        return loss

# Define the dataset class for 3D data
class ContrastiveImagingAndTabularDataset(Dataset):
    def __init__(self, image_data: np.ndarray, tabular_data: np.ndarray):
        self.image_data = torch.tensor(image_data, dtype=torch.float32)  # 3D data
        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.image_data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.image_data[index].unsqueeze(0)  # Add channel dimension
        tabular = self.tabular_data[index]
        return image, tabular

# Contrastive Image Tabular Pre-training(CITP)
def main():
    # Dummy data
    num_samples = 100
    image_data = np.random.rand(num_samples, 96, 96, 96)  # Replace with actual 3D image data
    tabular_data = np.random.rand(num_samples, 10)  # Replace with actual tabular data

    # Create dataset and dataloader
    dataset = ContrastiveImagingAndTabularDataset(image_data, tabular_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Adjust batch size as needed

    # Initialize models
    hidden_dim = 128  # Desired output dimension
    image_encoder = ContrastiveImageEncoder(hidden_dim)
    tabular_encoder = ContrastiveTabularEncoder(10, hidden_dim)  # Adjust input_dim based on actual data

    # Loss and optimizer
    contrastive_loss = ContrastiveLoss()
    optimizer = optim.Adam(list(image_encoder.parameters()) + list(tabular_encoder.parameters()), lr=0.001)

    # Training loop
    for epoch in range(10):  # Adjust the number of epochs as needed
        for images, tabulars in dataloader:
            optimizer.zero_grad()
            
            image_features = image_encoder(images)
            tabular_features = tabular_encoder(tabulars)
            
            # Contrastive loss
            loss = contrastive_loss(image_features, tabular_features)
            loss.backward()
            optimizer.step()
            
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    # Save the models
    torch.save(image_encoder.state_dict(), 'image_encoder.pth')
    torch.save(tabular_encoder.state_dict(), 'tabular_encoder.pth')
    
if __name__ == "__main__":
    main()