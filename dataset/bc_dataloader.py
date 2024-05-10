import pandas as pd
import os
from monai import transforms
from monai.data import DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# import torchvision.transforms as transforms
# from .feature_engineering import load_data
from .bc_feature_engineering import load_data

class CustomDataset(Dataset):
    def __init__(self, dataframe, mode='train'):
        self.dataframe = dataframe
        self.mode = mode
        
        # Define transforms for the data
        self.train_transform = transforms.Compose([
            transforms.LoadImage(image_only=True),
            transforms.AddChannel(),
            transforms.Resize(spatial_size=(96, 96, 96)),
            transforms.Orientation(axcodes="RAS"),
            transforms.ScaleIntensityRange(
                a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),  
            transforms.CropForeground(),
            transforms.RandFlip(prob=0.1, spatial_axis=0),
            transforms.RandFlip(prob=0.1, spatial_axis=1),
            transforms.RandFlip(prob=0.1, spatial_axis=2),
            transforms.RandRotate90(prob=0.1, max_k=3),
            transforms.RandScaleIntensity(factors=0.1, prob=0.1),
            transforms.RandShiftIntensity(offsets=0.1, prob=0.5),
            transforms.ToTensor(),  
        ])
        
        self.val_transform = transforms.Compose([   
            transforms.LoadImage(image_only=True),
            transforms.AddChannel(),
            transforms.Resize(spatial_size=(96, 96, 96)),
            transforms.Orientation(axcodes="RAS"),
            transforms.ScaleIntensityRange(
                a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),  
            # transforms.CropForeground(),
            transforms.ToTensor(),
        ])
        
        # Define transforms for the test data
        self.test_transform = transforms.Compose([   
            transforms.LoadImage(image_only=True),
            transforms.AddChannel(),
            # transforms.Spacingd(
            #     keys=["image"],
            #     pixdim=(1.5, 1.5, 2.0),
            #     mode=("bilinear"),
            # ),
            transforms.Resize(spatial_size=(96, 96, 96)),
            transforms.Orientation(axcodes="RAS"),
            transforms.ScaleIntensityRange(
                a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),   
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0] # 3 is the index of image_path in train_data
        
        # Apply transformations based on the mode
        if self.mode == 'train':
            image = self.train_transform(img_name)
        if self.mode == 'val':
            image = self.val_transform(img_name)            
        else:
            image = self.test_transform(img_name)
            
        Duct_8mm = self.dataframe.iloc[idx, 1] # 1 = index of Duct_diliatations_8mm in train_data
        Duct_10mm = self.dataframe.iloc[idx, 2] # 2 = index of Duct_diliatation_10mm in train_data
        vis_ct = self.dataframe.iloc[idx, 3]  # 3 = index of Visible_stone_CT in train_data
        pancreas = self.dataframe.iloc[idx, 4]  # 4 = index of Pancreatitis in train_data
        label = self.dataframe.iloc[idx, 5]  # 4 = index of target in train_data

        
                     
        # Convert to tensor and return all inputs
        Duct_8mm = torch.tensor(Duct_8mm, dtype=torch.float32)
        Duct_10mm = torch.tensor(Duct_10mm, dtype=torch.float32)
        vis_ct = torch.tensor(vis_ct, dtype=torch.float32)
        pancreas = torch.tensor(pancreas, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
               
        return image, Duct_8mm, Duct_10mm, vis_ct, pancreas, label
    
def getloader_bc(
    data_dir : 'str',
    excel_file : 'str',
    batch_size: int = 1,
    mode : str = "train",
    ):
    
    if mode == 'train':
        train_data, valid_data = load_data(data_dir, excel_file, mode='train')
        train_dataset = CustomDataset(train_data, mode='train')
        valid_dataset = CustomDataset(valid_data, mode='val')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader
    else:
        test_data = load_data(data_dir, excel_file, mode='test')
        test_dataset = CustomDataset(test_data, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        return test_loader



