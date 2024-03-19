import pandas as pd
from monai import transforms
from monai.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image
import torch
from .feature_engineering import load_data

class CustomDataset(Dataset):
    def __init__(self, dataframe, mode='train'):
        self.dataframe = dataframe
        self.mode = mode
        
        # Define transforms for the data
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=7)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)], p=0.5),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.5)], p=0.5),
            transforms.ToTensor(),  
        ])

        # Define transforms for the test data
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
        ])
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0] # 3 is the index of image_path in train_data
        image = Image.open(img_name)

        age = self.dataframe.iloc[idx, 1] # 1 = index of age_approx in train_data
        anatom_site = self.dataframe.iloc[idx, 2] # 2 = index of anatom_site_encoded in train_data
        sex = self.dataframe.iloc[idx, 3]  # 3 = index of sexes_encoded in train_data
        label = self.dataframe.iloc[idx, 4]  # 4 = index of target in train_data

        # Apply transformations based on the mode
        if self.mode == 'train':
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)
                     
        # Convert to tensor and return all inputs
        age = torch.tensor(age, dtype=torch.float32)
        anatom_site = torch.tensor(anatom_site, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
               
        return image, age, anatom_site, sex, label
    
def getloader(
    data_dir : 'str',
    batch_size: int = 1,
    mode : str = "train",
):
    if mode == 'train':
        train_data, valid_data = load_data(data_dir, mode='train')
        train_dataset = CustomDataset(train_data, mode='train')
        valid_dataset = CustomDataset(valid_data, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader
    else:
        test_data = load_data(data_dir, mode='test')
        test_dataset = CustomDataset(test_data, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        return test_loader

    

