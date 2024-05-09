from monai.data import DataLoader, Dataset, ImageDataset
from monai.transforms import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

import numpy as np
import pandas as pd
from glob import glob
import random
from tqdm import tqdm 
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    # PyTorch 시드 설정
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 가속화를 위해 cudnn을 사용하지 않음

# 시드 설정
set_seed(42)

class ImageBileductClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageBileductClassifier, self).__init__()
        # Load pre-trained backbone model
        # model = SwinUNETR(img_size=(96, 96, 96),in_channels=1,out_channels=1,feature_size=48)
        model = SwinUNETR(img_size=(96, 96, 96), in_channels=1,out_channels=1,feature_size=48)
        weight = torch.load("../model_swinvit.pt")
        model.load_from(weights=weight)
        # self.backbone_features = nn.Sequential(*list(model.children())[:-6])
        self.model = model

        # SwinUNetR의 out 레이어를 수정하여 출력 채널 수를 num_classes로 변경
        self.conv = nn.Conv3d(768, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1),bias=True)         
        # Classifier 추가를 위해 Global Average Pooling 및 Fully Connected 레이어 추가
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(1024, 256),
                                nn.Dropout(p=0.5),
                                nn.Linear(256, num_classes))
        

        
    def forward(self, image):
        # SwinUNetR의 forward 메서드 호출
        hidden_states_out = self.model.swinViT(image)     
        x = self.model.encoder1(image) # torch.Size([1, 48, 96, 96, 96])
        x = self.model.encoder2(hidden_states_out[0]) # torch.Size([1, 48, 48, 48, 48])
        x = self.model.encoder3(hidden_states_out[1]) # torch.Size([1, 96, 24, 24, 24])
        x = self.model.encoder4(hidden_states_out[2]) # torch.Size([1, 192, 12, 12, 12])
        x = self.model.encoder10(hidden_states_out[4]) # torch.Size([1, 768, 3, 3, 3])
        # SwinUNetR의 출력을 classifier에 통과시켜서 분류 작업 수행
        x = self.conv(x) # torch.Size([1, 1024, 3, 3, 3])
        x = self.global_avg_pool(x) # torch.Size([1, 1024, 1, 1, 1])
        x = x.view(x.size(0), -1) # torch.Size([1, 1024])
        # Fully connected layers for classification
        x = F.gelu(x)
        x = self.fc(x) 
        return x
    
# Define transforms for the test data
test_transform = Compose([   
    LoadImage(image_only=True),  # Load image only
    AddChannel(),  # Add channel dimension to image
    Resize(spatial_size=(96, 96, 96)),
    Orientation(axcodes="RAS"),
    ScaleIntensityRange(a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True),  # Scale intensity range
    ToTensor()  # Convert image to tensor
])

# Create a custom Dataset class to include labels
class CustomImageDataset(Dataset):
    def __init__(self, image_files, labels, transform):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = self.transform(image_path)  # Apply transform to image
        label = self.labels[idx]  # Get label corresponding to the image
        return image, label  # Return image and label

# Define the path to your CSV file
xlsx_file = "../datasets/Part1_nifti/bileduct_data_20240429a.xlsx"
df = pd.read_excel(xlsx_file, engine='openpyxl')
#Inclusion
df_inclusion = df[df['Inclusion']==1.0]
print('총 row : ', len(df_inclusion))
#DropNA
# na_count_by_column = df_inclusion.isna().sum()
# print(na_count_by_column)
df_inclusion.fillna(0.0,inplace=True)
#Column rename
df_inclusion.rename(columns={'환자번호': 'patient_id','Real_stone':'target' }, inplace=True)
#column select
columns = ['patient_id','Duct_diliatation', 'Visible_stone_CT', 'Pancreatitis','target']
data = df_inclusion[columns]
data['patient_id'] = data['patient_id'].astype(str)

image_list = sorted(glob("/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part1_nifti/test/*.nii.gz"))

# Initialize lists to store data
image_paths, Duct_diliatations, Visible_stone_CTs, Pancreatitis_values, targets= [],[],[],[],[]

for i in image_list:
    image_number = i.split('/')[-1].split('_')[0]
    if len(data[data['patient_id']==image_number]) > 0:
        Duct_diliatation = data.loc[data['patient_id'].str.startswith(image_number), 'Duct_diliatation'].values[0]
        Visible_stone_CT = data.loc[data['patient_id'].str.startswith(image_number), 'Visible_stone_CT'].values[0]
        Pancreatitis = data.loc[data['patient_id'].str.startswith(image_number), 'Pancreatitis'].values[0]
        target = data.loc[data['patient_id'].str.startswith(image_number), 'target'].values[0]

        # Append data to lists
        image_paths.append(i)
        Duct_diliatations.append(Duct_diliatation)
        Visible_stone_CTs.append(Visible_stone_CT)
        Pancreatitis_values.append(Pancreatitis)
        targets.append(target)

# Create a dictionary from lists
data_dict = {
    'image_path': image_paths,
    'Duct_diliatation': Duct_diliatations,
    'Visible_stone_CT': Visible_stone_CTs,
    'Pancreatitis': Pancreatitis_values,
    'target': targets
}

# Create a DataFrame from the dictionary
test_df = pd.DataFrame(data_dict)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the dataset using the custom Dataset class
test_ds = CustomImageDataset(image_files=test_df['image_path'], labels=test_df['target'], transform=test_transform)
# Create the DataLoader
test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)

model = ImageBileductClassifier(num_classes=1).to(device)

model.eval()
with torch.cuda.amp.autocast():    
    for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        image, label = batch[0].to(device), batch[1].to(device)
        output = model(image)
        # output =torch.softmax(output, 1).detach().cpu().numpy()
        # output = np.argmax(output, axis=1).astype(np.uint8)[0]
        print(output, label)
