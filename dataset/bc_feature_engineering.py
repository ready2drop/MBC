import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from glob import glob
import os


def load_data(data_dir : str, 
              excel_file : str,
                mode : str = "train",
                modality : str = 'mm',
                ):
    
    
    print("--------------Load RawData--------------")
    df = pd.read_excel(os.path.join(data_dir, excel_file), engine='openpyxl')

    #Inclusion
    print("--------------Inclusion--------------")
    df = df[df['Inclusion']==1.0]
    print('Total : ', len(df))

    print("--------------fillNA--------------")
    # data = data.dropna()
    df.fillna(0.0,inplace=True)
    print(df['Real_stone'].value_counts())

    #Column rename
    df.rename(columns={'환자번호': 'patient_id','Real_stone':'target' }, inplace=True)
    #column select
    columns = ['patient_id','Duct_diliatation_8mm', 'Duct_diliatation_10mm', 'Visible_stone_CT', 'Pancreatitis','target']
    data = df[columns]
    data['patient_id'] = data['patient_id'].astype(str)

    image_list = sorted(glob(os.path.join(data_dir,"*.nii.gz")))

    # Initialize lists to store data
    image_paths, Duct_diliatations_8mms, Duct_diliatations_10mms, Visible_stone_CTs, Pancreatitis_values, targets= [],[],[],[],[],[]

    for i in image_list:
        image_number = i.split('/')[-1].split('_')[0]
        if len(data[data['patient_id']==image_number]) > 0:
            Duct_diliatations_8mm = data.loc[data['patient_id'].str.startswith(image_number), 'Duct_diliatation_8mm'].values[0]
            Duct_diliatations_10mm = data.loc[data['patient_id'].str.startswith(image_number), 'Duct_diliatation_10mm'].values[0]
            Visible_stone_CT = data.loc[data['patient_id'].str.startswith(image_number), 'Visible_stone_CT'].values[0]
            Pancreatitis = data.loc[data['patient_id'].str.startswith(image_number), 'Pancreatitis'].values[0]
            target = data.loc[data['patient_id'].str.startswith(image_number), 'target'].values[0]

            # Append data to lists
            image_paths.append(i)
            Duct_diliatations_8mms.append(Duct_diliatations_8mm)
            Duct_diliatations_10mms.append(Duct_diliatations_10mm)
            Visible_stone_CTs.append(Visible_stone_CT)
            Pancreatitis_values.append(Pancreatitis)
            targets.append(target)

    # Create a dictionary from lists
    if modality == 'mm':
        data_dict = {
        'image_path': image_paths,
        'Duct_diliatations_8mm': Duct_diliatations_8mms,
        'Duct_diliatation_10mm': Duct_diliatations_10mms,
        'Visible_stone_CT': Visible_stone_CTs,
        'Pancreatitis': Pancreatitis_values,
        'target': targets
        }
    elif modality == 'image':
        data_dict = {
        'image_path': image_paths,
        'target': targets
        }
    else:
        raise AssertionError("Feature enginnering error")

    # Create a DataFrame from the dictionary
    train_df = pd.DataFrame(data_dict)
    
                
    
    
    print("--------------Class balance--------------")
    majority_class = train_df[train_df['target'] == 1.0]
    minority_class = train_df[train_df['target'] == 0.0]

    # Undersample the majority class to match the number of '1's in the minority class
    undersampled_majority_class = resample(majority_class,
                                        replace=False,
                                        n_samples=len(minority_class),
                                        random_state=42)

    # Concatenate minority class and undersampled majority class
    data = pd.concat([undersampled_majority_class, minority_class])
    print(data['target'].value_counts())
    
    
    
    # Splitting the test set into validation and test sets (70% train 20% validation, 10% test)
    train_data, test_data = train_test_split(data, test_size=0.3, stratify=data['target'], random_state=123)
    valid_data, test_data = train_test_split(test_data, test_size=0.4, stratify=test_data['target'], random_state=123)

        
    if mode == 'train':
        # Printing the shapes of the resulting datasets
        print("Train set shape:", train_data.shape)
        print("Validation set shape:", valid_data.shape)
        return train_data, valid_data
    else:
        print("Test set shape:", test_data.shape)
        return test_data
    