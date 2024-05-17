import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from monai.networks import nets

class ImagebileductClassifier_2d(nn.Module):
    def __init__(self, dict):
        self.num_classes = dict['num_classes']
        self.num_features = dict['num_features']
        self.model_architecture = dict['model_architecture']
        self.model_parameters = dict['model_parameters']
        super(ImagebileductClassifier_2d, self).__init__()
        
        # Load pre-trained backbone model
        backbone_model = getattr(models, self.model_architecture)(**self.model_parameters)
        
        # Remove the last classifier layer
        self.backbone_features = nn.Sequential(*list(backbone_model.children())[:-1])
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(self.num_features, 256)
        self.fc2 = nn.Linear(256, self.num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, image):
        # Forward pass through the pre-trained model
        image_features = self.backbone_features(image)
        image_features = F.avg_pool2d(image_features, image_features.size()[2:]).view(image.size(0), -1)  # Flatten
             
        # Fully connected layers for classification
        image_features = F.relu(self.fc1(image_features))
        image_features = self.dropout(image_features)
        output = self.fc2(image_features) 
        
        return output
    
class ImagebileductClassifier_3d(nn.Module):
    def __init__(self, dict):
        self.num_classes = dict['num_classes']
        self.num_features = dict['num_features']
        self.model_architecture = dict['model_architecture']
        self.model_parameters = dict['model_parameters']
        self.pretrain_path = dict['pretrain_path']
        super(ImagebileductClassifier_3d, self).__init__()
        
        # Load pre-trained backbone model
        model = getattr(nets, self.model_architecture)(**self.model_parameters)
        weight = torch.load(self.pretrain_path)
        model.load_from(weights=weight)
        self.model = model
        
        # Classifier 추가를 위해 Global Average Pooling 및 Fully Connected 레이어 추가
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(self.num_features, 256),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(256, self.num_classes))
        

        
    def forward(self, image):
        # SwinUNetR의 forward 메서드 호출
        hidden_states_out = self.model.swinViT(image)     
        x = self.model.encoder1(image) # torch.Size([1, 48, 96, 96, 96])
        x = self.model.encoder2(hidden_states_out[0]) # torch.Size([1, 48, 48, 48, 48])
        x = self.model.encoder3(hidden_states_out[1]) # torch.Size([1, 96, 24, 24, 24])
        x = self.model.encoder4(hidden_states_out[2]) # torch.Size([1, 192, 12, 12, 12])
        x = self.model.encoder10(hidden_states_out[4]) # torch.Size([1, 768, 3, 3, 3])
        # SwinUNetR의 출력을 classifier에 통과시켜서 분류 작업 수행
        x = self.global_avg_pool(x) # torch.Size([1, 768, 1, 1, 1])
        x = x.view(x.size(0), -1) # torch.Size([1, 1024])
        
        # Fully connected layers for classification
        x = self.fc(x)
        
        return x