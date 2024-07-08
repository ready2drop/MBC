import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from glob import glob
import os


def load_data(data_dir : str, 
              excel_file : str,
                mode : str = "train",
                modality : str = 'mm',
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
    columns = ['patient_id', 'EDTA_Hb', 'EDTA_PLT', 'EDTA_WBC', 'SST_ALP', 'SST_ALT',
       'SST_AST', 'SST_CRP', 'FIRST_SBP', 'FIRST_DBP', 'FIRST_HR', 'FIRST_RR',
       'FIRST_BT', 'VISIBLE_STONE_CT', 'PANCREATITIS','SEX', 'AGE',
        'DUCT_DILIATATION_8MM', 'DUCT_DILIATATION_10MM','target']
    # columns = ['patient_id', 'EDTA_Hb', 'EDTA_PLT', 'EDTA_WBC', 'SST_ALP', 'SST_ALT',
    #    'SST_AST', 'SST_CRP', 'FIRST_SBP', 'FIRST_DBP', 'FIRST_HR', 'FIRST_RR',
    #    'FIRST_BT', 'PANCREATITIS','SEX', 'AGE',
    #     'DUCT_DILIATATION_8MM', 'DUCT_DILIATATION_10MM','target']
    
    data = df[columns]
    data['patient_id'] = data['patient_id'].astype(str)

    image_list = sorted(glob(os.path.join(data_dir,"*.nii.gz")))

    def get_patient_data(image_number):
        row = data[data['patient_id'].astype(str).str.startswith(image_number)]
        return row.iloc[0, 1:].tolist() if not row.empty else None
    
    # Rename column 
    data_dict = {key: [] for key in ['image_path','EDTA_Hb', 'EDTA_PLT', 'EDTA_WBC', 'SST_ALP', 'SST_ALT',
       'SST_AST', 'SST_CRP', 'FIRST_SBP', 'FIRST_DBP', 'FIRST_HR', 'FIRST_RR',
       'FIRST_BT', 'VISIBLE_STONE_CT', 'PANCREATITIS','SEX', 'AGE',
        'DUCT_DILIATATION_8MM', 'DUCT_DILIATATION_10MM','target']}
    # data_dict = {key: [] for key in ['image_path','EDTA_Hb', 'EDTA_PLT', 'EDTA_WBC', 'SST_ALP', 'SST_ALT',
    #    'SST_AST', 'SST_CRP', 'FIRST_SBP', 'FIRST_DBP', 'FIRST_HR', 'FIRST_RR',
    #    'FIRST_BT', 'PANCREATITIS','SEX', 'AGE',
    #     'DUCT_DILIATATION_8MM', 'DUCT_DILIATATION_10MM','target']}


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
        columns_to_scale = ['EDTA_Hb', 'EDTA_PLT', 'EDTA_WBC', 'SST_ALP', 'SST_ALT',
       'SST_AST', 'SST_CRP', 'FIRST_SBP', 'FIRST_DBP', 'FIRST_HR', 'FIRST_RR',
       'FIRST_BT','AGE']
        scaler = MinMaxScaler()
        train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])

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
    # data = train_df
    print(data['target'].value_counts())
    train_data, test_data = train_test_split(data, test_size=0.3, stratify=data['target'], random_state=123)
    valid_data, test_data = train_test_split(test_data, test_size=0.4, stratify=test_data['target'], random_state=123)
    
    # if modality == 'mm':
    #     ## oversampling
    #     image_paths = train_df['image_path']
    #     X = train_df.drop(['image_path','target'], axis=1)
    #     y = train_df['target']

    #     # Apply SMOTE
    #     smote = SMOTE(random_state=42)
    #     X_res, y_res = smote.fit_resample(X, y)

    #     # Combine the resampled features and target into a single DataFrame
    #     train_df_resampled = pd.concat([X_res, y_res], axis=1)
    #     num_samples_to_add = len(train_df_resampled) - len(image_paths)
    #     repeated_image_paths = pd.concat([image_paths, image_paths.sample(num_samples_to_add, replace=True)]).reset_index(drop=True)

    #     train_df_resampled.insert(0, 'image_path', repeated_image_paths)
    #     print(train_df_resampled['target'].value_counts())

    # print("--------------Scaling--------------")
    # if modality == 'mm':
    #     # MinMaxScaler 객체 생성
    #     columns_to_scale = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'AGE', 'blood_test']
    #     scaler = MinMaxScaler()
    #     train_df_resampled[columns_to_scale] = scaler.fit_transform(train_df_resampled[columns_to_scale])

    #     # Splitting the test set into validation and test sets (70% train 20% validation, 10% test)
    #     train_data, test_data = train_test_split(train_df_resampled, test_size=0.3, stratify=train_df_resampled['target'], random_state=123)
    #     valid_data, test_data = train_test_split(test_data, test_size=0.4, stratify=test_data['target'], random_state=123)
    # else: 
    #     # Splitting the test set into validation and test sets (70% train 20% validation, 10% test)
    #     train_data, test_data = train_test_split(data, test_size=0.3, stratify=data['target'], random_state=123)
    #     valid_data, test_data = train_test_split(test_data, test_size=0.4, stratify=test_data['target'], random_state=123)


        
    if mode == 'train':
        print("Train set shape:", train_data.shape)
        print("Validation set shape:", valid_data.shape)
        return train_data, valid_data

    elif mode == 'test':
        print("Test set shape:", test_data.shape)
        return test_data
    
    else:
        raise ValueError("Choose mode!")
    