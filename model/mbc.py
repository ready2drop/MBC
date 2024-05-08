import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class MultiModalbileductClassifier(nn.Module):
    def __init__(self, num_classes, num_features, model_architecture, model_parameters):
        super(MultiModalbileductClassifier, self).__init__()
        # Load pre-trained backbone model
        backbone_model = getattr(models, model_architecture)(**model_parameters)
        
        # Remove the last classifier layer
        self.backbone_features = nn.Sequential(*list(backbone_model.children())[:-1])
        
        # Define additional feature dimensions
        self.age_dim = 1  # assuming age is a single scalar value
        self.anatom_site_dim = 1  # assuming anatomical site is a single scalar value
        self.sex_dim = 1  # assuming sex is a single scalar value
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(num_features + self.age_dim + self.anatom_site_dim + self.sex_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
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