import torch
import torch.nn as nn
from monai.networks import nets


class ImageEncoder3D(nn.Module):
    def __init__(self, dict):
        self.output_dim = dict['output_dim']
        self.num_features = dict['num_features']
        self.model_architecture = dict['model_architecture']
        self.model_parameters = dict['model_parameters']
        self.pretrain_path = dict['image_pretrain_path']
        super(ImageEncoder3D, self).__init__()
        
        # Load pre-trained backbone model
        model = getattr(nets, self.model_architecture)(**self.model_parameters)
        weight = torch.load(self.pretrain_path)
        model.load_from(weights=weight)
        self.model = model
        
        # Classifier 추가를 위해 Global Average Pooling 및 Fully Connected 레이어 추가
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(self.num_features, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(256, self.output_dim),
                                )
 
                
    def forward(self, image):
        # SwinUNetR의 forward 메서드 호출
        hidden_states_out = self.model.swinViT(image)   
        x = self.model.encoder1(image) # torch.Size([1, 48, 96, 96, 96])
        x = self.model.encoder2(hidden_states_out[0]) # torch.Size([1, 48, 48, 48, 48])
        x = self.model.encoder3(hidden_states_out[1]) # torch.Size([1, 96, 24, 24, 24])
        x = self.model.encoder4(hidden_states_out[2]) # torch.Size([1, 192, 12, 12, 12])
        x = self.global_avg_pool(x) # torch.Size([1, 192, 1, 1, 1])
        x = x.view(x.size(0), -1) # torch.Size([1, 192])
        x = self.fc(x)
        
        return x