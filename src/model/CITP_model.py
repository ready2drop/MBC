import torch
import torch.nn as nn
from monai.networks import nets
from typing import Tuple

# Define the 3D Vision Transformer encoder
class ContrastiveImageEncoder(nn.Module):
    def __init__(self, dict):
        self.hidden_dim = dict['hidden_dim']
        self.num_features = dict['num_features']
        self.model_architecture = dict['model_architecture']
        self.model_parameters = dict['model_parameters']
        self.mode = dict['mode']
        
        super(ContrastiveImageEncoder, self).__init__()
        
        # Load pre-trained backbone model
        self.model = getattr(nets, self.model_architecture)(**self.model_parameters)

        self.fc = nn.Linear(self.num_features, self.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.model(x)  # torch.Size([batch, 216, 768])
        features = self.fc(features.mean(dim=1))  # torch.Size([batch, hidden_dim])
        
        return features


class ContrastiveTabularEncoder(nn.Module):
    def __init__(self, dict):
        self.input_dim = dict['input_dim']
        self.hidden_dim = dict['hidden_dim']
        
        super(ContrastiveTabularEncoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CITPModel(nn.Module):
    def __init__(self, dict, temperature: float = 0.07):
        super(CITPModel, self).__init__()
        self.image_encoder = ContrastiveImageEncoder(dict)
        self.tabular_encoder = ContrastiveTabularEncoder(dict)
        self.temperature = temperature
    
    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.image_encoder(image)
        tabular_features = self.tabular_encoder(tabular)
        return image_features, tabular_features
    
    def compute_contrastive_loss(self, image_features: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        image_features = nn.functional.normalize(image_features, dim=-1)
        tabular_features = nn.functional.normalize(tabular_features, dim=-1)
        
        similarity_matrix = torch.matmul(image_features, tabular_features.T) / self.temperature
        
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss