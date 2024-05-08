import pandas as pd
import os
from monai import transforms
from monai.data import DataLoader
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
)

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# import torchvision.transforms as transforms
from PIL import Image
import torch
from .feature_engineering import load_data

# class CustomDataset(Dataset):
#     def __init__(self, dataframe, mode='train'):
#         self.dataframe = dataframe
#         self.mode = mode
        
#         # Define transforms for the data
#         self.train_transform = transforms.Compose([
#             transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
#             transforms.ScaleIntensityRanged(
#                 keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
#             ),
#             transforms.CropForegroundd(
#                 keys=["image", "label"], source_key="image"
#             ),
#             transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
#             transforms.Spacingd(
#                 keys=["image", "label"],
#                 pixdim=(1.5, 1.5, 2.0),
#                 mode=("bilinear", "nearest"),
#             ),
#             transforms.RandScaleCropd(
#                 keys=["image", "label"], 
#                 roi_scale=[0.75, 0.85, 1.0],
#                 random_size=False
#             ),
#             transforms.RandCropByPosNegLabeld(
#                 keys=["image", "label"],
#                 label_key="label",
#                 spatial_size=(96, 96, 96),
#                 pos=1,
#                 neg=1,
#                 num_samples=4,
#                 image_key="image",
#                 image_threshold=0,
#             ),
            
#             transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
#             transforms.RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),

#             # transforms.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
#             transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
#             transforms.ToTensord(keys=["image", "label"]),  
#         ])
        
#         self.val_transform = transforms.Compose([   
#             transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
#             transforms.ScaleIntensityRanged(
#                 keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
#             ),
#             transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
#             transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
#             transforms.Spacingd(
#                 keys=["image", "label"],
#                 pixdim=(1.5, 1.5, 2.0),
#                 mode=("bilinear", "nearest"),
#             ),
#             # transforms.Resized(keys=["image", "label"], spatial_size=(spatial_size, image_size, image_size)),
#             transforms.ToTensord(keys=["image", "label"]),
#         ])
        
#         # Define transforms for the test data
#         self.test_transform = transforms.Compose([   
#             transforms.LoadImaged(keys=["image", "raw_label"]),
#             transforms.AddChanneld(keys=["image", "raw_label"]),
#             # transforms.Spacingd(
#             #     keys=["image"],
#             #     pixdim=(1.5, 1.5, 2.0),
#             #     mode=("bilinear"),
#             # ),
#             transforms.ScaleIntensityRanged(
#                 keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
#             ),   
#             transforms.ToTensord(keys=["image","raw_label"]),
#         ])
        
    # def __len__(self):
    #     return len(self.dataframe)

    # def __getitem__(self, idx):
    #     img_name = self.dataframe.iloc[idx, 0] # 3 is the index of image_path in train_data
    #     image = Image.open(img_name)

    #     age = self.dataframe.iloc[idx, 1] # 1 = index of age_approx in train_data
    #     anatom_site = self.dataframe.iloc[idx, 2] # 2 = index of anatom_site_encoded in train_data
    #     sex = self.dataframe.iloc[idx, 3]  # 3 = index of sexes_encoded in train_data
    #     label = self.dataframe.iloc[idx, 4]  # 4 = index of target in train_data

    #     # Apply transformations based on the mode
    #     if self.mode == 'train':
    #         image = self.train_transform(image)
    #     else:
    #         image = self.test_transform(image)
                     
    #     # Convert to tensor and return all inputs
    #     age = torch.tensor(age, dtype=torch.float32)
    #     anatom_site = torch.tensor(anatom_site, dtype=torch.float32)
    #     sex = torch.tensor(sex, dtype=torch.float32)
    #     label = torch.tensor(label, dtype=torch.long)
               
    #     return image, age, anatom_site, sex, label
    
def getloader_bc(
    data_dir : 'str',
    batch_size: int = 1,
    mode : str = "train",
):
    train_transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image"
            ),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            transforms.RandScaleCropd(
                keys=["image", "label"], 
                roi_scale=[0.75, 0.85, 1.0],
                random_size=False
            ),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            
            transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),

            # transforms.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            transforms.ToTensord(keys=["image", "label"]),  
        ])
        
    val_transform = transforms.Compose([   
            transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            # transforms.Resized(keys=["image", "label"], spatial_size=(spatial_size, image_size, image_size)),
            transforms.ToTensord(keys=["image", "label"]),
        ])
        
        # Define transforms for the test data
    test_transform = transforms.Compose([   
            transforms.LoadImaged(keys=["image", "raw_label"]),
            transforms.AddChanneld(keys=["image", "raw_label"]),
            # transforms.Spacingd(
            #     keys=["image"],
            #     pixdim=(1.5, 1.5, 2.0),
            #     mode=("bilinear"),
            # ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),   
            transforms.ToTensord(keys=["image","raw_label"]),
        ])
    # if mode == 'train':
    #     train_data, valid_data = load_data(data_dir, mode='train')
    #     train_dataset = CustomDataset(train_data, mode='train')
    #     valid_dataset = CustomDataset(valid_data, mode='train')
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    #     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    #     return train_loader, valid_loader
    # else:
    #     test_data = load_data(data_dir, mode='test')
    #     test_dataset = CustomDataset(test_data, mode='test')
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    #     return test_loader
    if mode == 'train':
        train_data = load_decathlon_datalist(os.path.join(data_dir, "dataset.json"), False, 'training')
        val_data = load_decathlon_datalist(os.path.join(data_dir, "dataset.json"), False, 'validation')
        train_dataset = CacheDataset(
            data=train_data,
            transform=train_transform,
            cache_num=len(train_data),
            cache_rate=1.0,
            num_workers=max(2, 16),
        )
        val_dataset = CacheDataset(
                data=val_data,
                transform=val_transform,
                cache_num=len(val_data),
                cache_rate=1.0,
                num_workers=max(2, 16),
        )
        train_dataloader = ThreadDataLoader(
                dataset=train_dataset,
                num_workers=max(2, 16),
                batch_size=batch_size, 
                shuffle=True 
            )
        val_dataloader = ThreadDataLoader(
                    dataset=val_dataset,
                    num_workers=max(2, 20),
                    batch_size=batch_size, 
                    shuffle=True 
                )
        loader = [train_dataloader, val_dataloader]

    else:
        test_data = load_decathlon_datalist(os.path.join(data_dir, "dataset.json"), False, 'test')
        test_dataset = CacheDataset(
                data=test_data,
                transform=test_transform,
                cache_num=len(test_data),
                cache_rate=1.0,
                num_workers=max(2, 16),
            )
    
        test_dataloader = ThreadDataLoader(
                    dataset=test_dataset,
                    num_workers=max(2, 16),
                    batch_size=batch_size, 
                    shuffle=True 
                )

        loader = test_dataloader

    return loader
    

