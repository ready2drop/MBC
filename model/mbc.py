import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from monai.networks import nets

class MultiModalbileductClassifier_2d(nn.Module):
    def __init__(self,dict):
        self.num_classes = dict['num_classes']
        self.num_features = dict['num_features']
        self.model_architecture = dict['model_architecture']
        self.model_parameters = dict['model_parameters']
        super(MultiModalbileductClassifier_2d, self).__init__()
        
        # Load pre-trained backbone model
        backbone_model = getattr(models, self.model_architecture)(**self.model_parameters)
        
        # Remove the last classifier layer
        self.backbone_features = nn.Sequential(*list(backbone_model.children())[:-1])
        
        # Define additional feature dimensions
        self.age_dim = 1  # assuming age is a single scalar value
        self.anatom_site_dim = 1  # assuming anatomical site is a single scalar value
        self.sex_dim = 1  # assuming sex is a single scalar value
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(self.num_features + self.age_dim + self.anatom_site_dim + self.sex_dim, 256)
        self.fc2 = nn.Linear(256, self.num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, image, age, anatom_site, sex):
        # Forward pass through the pre-trained model
        image_features = self.backbone_features(image)
        image_features = F.avg_pool2d(image_features, image_features.size()[2:]).view(image.size(0), -1)  # Flatten
        
        # Reshape age, anatom_site, and sex tensors
        age = age.view(-1, 1)  # Reshape to [batch_size, 1]
        anatom_site = anatom_site.view(-1, 1)  # Reshape to [batch_size, 1]
        sex = sex.view(-1, 1)  # Reshape to [batch_size, 1]
        # Concatenate image features with additional features
        additional_features = torch.cat((age, anatom_site, sex), dim=1)
        combined_features = torch.cat((image_features, additional_features), dim=1)
        
        # Fully connected layers for classification
        combined_features = F.relu(self.fc1(combined_features))
        combined_features = self.dropout(combined_features)
        output = self.fc2(combined_features) 
        
        return output
    
class MultiModalbileductClassifier_3d(nn.Module):
    def __init__(self, dict):
        self.num_classes = dict['num_classes']
        self.num_features = dict['num_features']
        self.model_architecture = dict['model_architecture']
        self.model_parameters = dict['model_parameters']
        self.pretrain_path = dict['pretrain_path']
        super(MultiModalbileductClassifier_3d, self).__init__()
        
        # Load pre-trained backbone model
        model = getattr(nets, self.model_architecture)(**self.model_parameters)
        weight = torch.load(self.pretrain_path)
        model.load_from(weights=weight)
        self.model = model
        self.feature_dim = 4  # Duct_diliatations_8mm, Duct_diliatation_10mm, Visible_stone_CT, Pancreatitis
        # Classifier 추가를 위해 Global Average Pooling 및 Fully Connected 레이어 추가
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(self.num_features, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(256, self.num_classes),
                                )
        

        
    def forward(self, image, Duct_diliatations_8mm, Duct_diliatation_10mm, Visible_stone_CT, Pancreatitis):
        # SwinUNetR의 forward 메서드 호출
        hidden_states_out = self.model.swinViT(image)     
        x = self.model.encoder1(image) # torch.Size([1, 48, 96, 96, 96])
        x = self.model.encoder2(hidden_states_out[0]) # torch.Size([1, 48, 48, 48, 48])
        x = self.model.encoder3(hidden_states_out[1]) # torch.Size([1, 96, 24, 24, 24])
        x = self.model.encoder4(hidden_states_out[2]) # torch.Size([1, 192, 12, 12, 12])
        # x = self.model.encoder10(hidden_states_out[4]) # torch.Size([1, 768, 3, 3, 3])
        # SwinUNetR의 출력을 classifier에 통과시켜서 분류 작업 수행
        x = self.global_avg_pool(x) # torch.Size([1, 768, 1, 1, 1])
        x = x.view(x.size(0), -1) # torch.Size([1, 1024])
        
        # Reshape age, anatom_site, and sex tensors
        Duct_diliatations_8mm = Duct_diliatations_8mm.view(-1, 1)  # Reshape to [batch_size, 1]
        Duct_diliatation_10mm = Duct_diliatation_10mm.view(-1, 1)  # Reshape to [batch_size, 1]
        Visible_stone_CT = Visible_stone_CT.view(-1, 1)  # Reshape to [batch_size, 1]
        Pancreatitis = Pancreatitis.view(-1, 1)  # Reshape to [batch_size, 1]

        # Concatenate image features with additional features
        additional_features = torch.cat((Duct_diliatations_8mm, Duct_diliatation_10mm, Visible_stone_CT, Pancreatitis), dim=1)
        combined_features = torch.cat((x, additional_features), dim=1)
        
        # Fully connected layers for classification
        x = self.fc(combined_features)
        
        return x