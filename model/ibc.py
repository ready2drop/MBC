import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


class ImageBileductClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageBileductClassifier, self).__init__()
        # Load pre-trained backbone model
        model = SwinUNETR(img_size=(96, 96, 96),in_channels=1,out_channels=14,feature_size=48)
        weight = torch.load("../model_swinvit.pt")
        model.load_from(weights=weight)
        self.backbone_features = nn.Sequential(*list(model.children())[:-6])
        # SwinUNetR의 out 레이어를 수정하여 출력 채널 수를 num_classes로 변경
        self.conv = nn.Conv3d(768, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))         
        # Classifier 추가를 위해 Global Average Pooling 및 Fully Connected 레이어 추가
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, image):
        # SwinUNetR의 forward 메서드 호출
        x = self.backbone_features(image)
        # SwinUNetR의 출력을 classifier에 통과시켜서 분류 작업 수행
        x = self.conv(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        # Fully connected layers for classification
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x