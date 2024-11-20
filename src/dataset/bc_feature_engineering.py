import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from glob import glob
import os
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


def load_data(data_dir : str, 
              excel_file : str,
                mode : str = "train",
                modality : str = 'mm',
                phase : str = 'portal',  # 'portal', 'pre-enhance', 'combine'
                smote = bool,
                ):
    
    
    print("--------------Load RawData--------------")
    df = pd.read_csv(os.path.join(data_dir, excel_file))
    
    #Inclusion
    print("--------------Inclusion--------------")
    print('Total : ', len(df))

    print("--------------fillNA--------------")
    # data = data.dropna()
    df.fillna(0.0,inplace=True)
    print(df['REAL_STONE'].value_counts())

    #Column rename
    df.rename(columns={'ID': 'patient_id', 'REAL_STONE':'target'}, inplace=True)
    
    #Column select
    columns = ['patient_id', 'Hb', 'PLT', 'WBC', 'ALP', 'ALT',
       'AST', 'CRP', 'BILIRUBIN', 'FIRST_SBP', 'FIRST_DBP', 'FIRST_HR', 'FIRST_RR',
       'FIRST_BT', 'VISIBLE_STONE_CT', 'PANCREATITIS','SEX', 'AGE',
        'DUCT_DILIATATION_8MM', 'DUCT_DILIATATION_10MM','target']
    # columns = ['patient_id', 'Hb', 'PLT', 'WBC', 'ALP', 'ALT',
    #    'AST', 'CRP', 'BILIRUBIN', 'FIRST_SBP', 'FIRST_DBP', 'FIRST_HR', 'FIRST_RR',
    #    'FIRST_BT', 'PANCREATITIS','SEX', 'AGE',
    #     'DUCT_DILIATATION_8MM', 'DUCT_DILIATATION_10MM','target']
    
    data = df[columns]
    data['patient_id'] = data['patient_id'].astype(str)

    image_list = sorted(glob(os.path.join(data_dir,"*.nii.gz")))

    def get_patient_data(image_number):
        row = data[data['patient_id'].astype(str).str.startswith(image_number)]
        return row.iloc[0, 1:].tolist() if not row.empty else None
    
    # Rename column 
    data_dict = {key: [] for key in ['image_path','Hb', 'PLT', 'WBC', 'ALP', 'ALT',
       'AST', 'CRP', 'BILIRUBIN', 'FIRST_SBP', 'FIRST_DBP', 'FIRST_HR', 'FIRST_RR',
       'FIRST_BT', 'VISIBLE_STONE_CT', 'PANCREATITIS','SEX', 'AGE',
        'DUCT_DILIATATION_8MM', 'DUCT_DILIATATION_10MM','target']}
    # data_dict = {key: [] for key in ['image_path','Hb', 'PLT', 'WBC', 'ALP', 'ALT',
    #    'AST', 'CRP', 'BILIRUBIN', 'FIRST_SBP', 'FIRST_DBP', 'FIRST_HR', 'FIRST_RR',
    #    'FIRST_BT', 'PANCREATITIS','SEX', 'AGE',
    #     'DUCT_DILIATATION_8MM', 'DUCT_DILIATATION_10MM','target']}

    # Filter images based on the phase
    if phase == 'portal':
        # Filter the images for the 'portal' phase by checking for 'Portal' in the filename
        image_list = [img for img in image_list if 'Portal' in os.path.basename(img)]
    elif phase == 'pre-enhance':
        # Filter the images for the 'pre-enhance' phase by checking for 'Pre_enhance' in the filename
        image_list = [img for img in image_list if 'Pre_enhance' in os.path.basename(img)]
    elif phase == 'combine':
        # Include both 'portal' and 'pre-enhance' images for the 'combine' phase
        portal_images = [img for img in image_list if 'Portal' in os.path.basename(img)]
        pre_enhance_images = [img for img in image_list if 'Pre_enhance' in os.path.basename(img)]
        image_list = portal_images + pre_enhance_images
    else:
        raise ValueError("Invalid phase. Choose from ['portal', 'pre-enhance', 'combine']")


    for image_path in image_list:
        image_number = os.path.basename(image_path).split('_')[0]
        patient_data = get_patient_data(image_number)
        if patient_data:
            data_dict['image_path'].append(image_path)
            keys_list = list(data_dict.keys())[1:]
            for key, value in zip(keys_list, patient_data):
                if key == 'image_path':
                    continue
                data_dict[key].append(value)

    if modality == 'image':
            data_dict = {k: data_dict[k] for k in ['image_path', 'target']}
            
    elif modality not in ['mm', 'tabular']:
        raise AssertionError("Select Modality for Feature engineering!")

    #Create a DataFrame from the dictionary
    train_df = pd.DataFrame(data_dict)
    
    #if only  tabular use 
    if modality == 'tabular':
        train_df = data
        
    print("--------------Scaling--------------")
    if modality in ['mm', 'tabular']:
        columns_to_scale = ['Hb', 'PLT', 'WBC', 'ALP', 'ALT',
       'AST', 'CRP', 'BILIRUBIN', 'FIRST_SBP', 'FIRST_DBP', 'FIRST_HR', 'FIRST_RR',
       'FIRST_BT','AGE']
        scaler = MinMaxScaler()
        train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])

    if mode == 'train' or mode == 'test':
        print("--------------Class balance--------------")
        # undersampling
        majority_class = train_df[train_df['target'] == 1.0]
        minority_class = train_df[train_df['target'] == 0.0]

        # Undersample the majority class to match the number of '1's in the minority class
        undersampled_majority_class = resample(majority_class,
                                            replace=False,
                                            n_samples=len(minority_class),
                                            random_state=42)

        # Concatenate minority class and undersampled majority class
        data = pd.concat([undersampled_majority_class, minority_class])
        
        # print("--------------Class imbalance--------------")
        if smote:  # Apply SMOTE if the flag is set
            data = train_df
            print(data['target'].value_counts())
            print("Applying SMOTE...")
            smote = SMOTE(sampling_strategy='all', random_state=42)
            X_data = data.drop(columns=['target'])
            y_data = data['target']
            X_data_res, y_data_res = smote.fit_resample(X_data, y_data)
            data_resampled = pd.DataFrame(X_data_res, columns=X_data.columns)
            data_resampled['target'] = y_data_res
            data = data_resampled  # Update train_data with resampled data
            print(data['target'].value_counts())
            
        train_data, test_data = train_test_split(data, test_size=0.3, stratify=data['target'], random_state=123)
        valid_data, test_data = train_test_split(test_data, test_size=0.4, stratify=test_data['target'], random_state=123)
        
        if mode == 'train':
            print("Train set shape:", train_data.shape)
            print("Validation set shape:", valid_data.shape)
            return train_data, valid_data

        elif mode == 'test':
            print("Test set shape:", test_data.shape)
            return test_data
        
    elif mode == 'pretrain' or mode == 'eval':
        pretrain_data, eval_data = train_test_split(train_df, test_size=0.1, random_state=123)
        if mode == 'pretrain':
            print("Pretrain set shape:", pretrain_data.shape)
            return pretrain_data
        elif mode == 'eval':
            print("Validation set shape:", eval_data.shape)
            return eval_data
    
    else:
        raise ValueError("Choose mode!")
    