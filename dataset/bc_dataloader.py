import pandas as pd
import os
import torch
from torch.utils.data import Dataset

from monai import transforms
from monai.data import DataLoader
from .bc_feature_engineering import load_data

class CustomDataset(Dataset):
    def __init__(self, dataframe, modality, mode):
        self.dataframe = dataframe
        self.modality = modality
        self.mode = mode
        
        # Define transforms for the data
        self.train_transform = transforms.Compose([
            transforms.LoadImage(image_only=True),
            transforms.AddChannel(),
            transforms.Orientation(axcodes="RAS"),
            transforms.ScaleIntensityRange(
                a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),  
            # transforms.CropForeground(),
            # transforms.SpatialCrop(roi_center=(300, 250, 0), roi_size=(256,256,1000)),
            transforms.Spacing(
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            # transforms.Zoom(zoom=1.2),
            transforms.Resize(spatial_size=(96, 96, 96)),
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
            transforms.Orientation(axcodes="RAS"),
            transforms.ScaleIntensityRange(
                a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),  
            # transforms.SpatialCrop(roi_center=(300, 250, 0), roi_size=(256,256,1000)),
            transforms.Spacing(
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            # transforms.CropForeground(),
            transforms.Resize(spatial_size=(96, 96, 96)),
            transforms.ToTensor(),
        ])
        
        # Define transforms for the test data
        self.test_transform = transforms.Compose([   
            transforms.LoadImage(image_only=True),
            transforms.AddChannel(),
            transforms.Orientation(axcodes="RAS"),
            transforms.ScaleIntensityRange(
                a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),   
            transforms.Spacing(
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            transforms.CropForeground(),
            # transforms.SpatialCrop(roi_center=(200, 200, 0), roi_size=(256,256,1000)),
            transforms.Resize(spatial_size=(96, 96, 96)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
                        
        if self.modality == 'mm':
            # Apply transformations based on the mode
            img_name = self.dataframe.iloc[idx, 0] # 0 is the index of image_path in train_data

            # Check the shape of the image and handle if it doesn't meet the criteria
            image = transforms.LoadImage(image_only=True)(img_name)
            if image.shape[0] != 512 or image.shape[1] != 512 or image.shape[2] is None:
                # print(f"Skipping image {img_name} due to incorrect shape: {image.shape}")
                idx = (idx + 1) % len(self.dataframe)
                img_name = self.dataframe.iloc[idx, 0]

            
            if self.mode == 'train':
                image = self.train_transform(img_name)
            elif self.mode == 'val':
                image = self.val_transform(img_name)            
            else:
                image = self.test_transform(img_name)
            
            meta_dict = image.meta
            
            data = self.dataframe.iloc[idx, 1:-1].values.astype(float)
            features = torch.tensor(data, dtype=torch.float32)
            label = torch.tensor(self.dataframe.iloc[idx, -1], dtype=torch.long)
            
            return image, features, label, meta_dict
            
        elif self.modality == 'image':
            # Apply transformations based on the mode
            img_name = self.dataframe.iloc[idx, 0] # 0 is the index of image_path in train_data

            # Check the shape of the image and handle if it doesn't meet the criteria
            image = transforms.LoadImage(image_only=True)(img_name)
            if image.shape[0] != 512 or image.shape[1] != 512 or image.shape[2] is None:
                # print(f"Skipping image {img_name} due to incorrect shape: {image.shape}")
                idx = (idx + 1) % len(self.dataframe)
                img_name = self.dataframe.iloc[idx, 0]

            
            if self.mode == 'train':
                image = self.train_transform(img_name)
            elif self.mode == 'val':
                image = self.val_transform(img_name)            
            else:
                image = self.test_transform(img_name)
            
            meta_dict = image.meta
            
            label = self.dataframe.iloc[idx, 1]
            label = torch.tensor(label, dtype=torch.long)
            
            return image, label, meta_dict
        
        elif self.modality == 'tabular':
            data = self.dataframe.iloc[idx, 1:-1].values.astype(float)
            features = [torch.tensor(data[i], dtype=torch.float32) for i in range(len(data))]
            label = torch.tensor(self.dataframe.iloc[idx, -1], dtype=torch.long)
            
            return features, label
            
        else:
            raise AssertionError("Data loader error")
        
     
    
def getloader_bc(
    data_dir : 'str',
    excel_file : 'str',
    batch_size: int = 1,
    mode : str = "train",
    modality : str = "mm",
    ):
    
    if mode == 'train':
        train_data, valid_data = load_data(data_dir, excel_file, mode, modality)
        train_dataset = CustomDataset(train_data, modality, mode='train')
        valid_dataset = CustomDataset(valid_data, modality, mode='val')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader
    elif mode == 'test':
        test_data = load_data(data_dir, excel_file, mode, modality)
        test_dataset = CustomDataset(test_data, modality, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        return test_loader
    else:
        raise ValueError("Choose mode!")



