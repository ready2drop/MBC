import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
    if excel_file.split('.')[-1]=='xlsx': 
        df = pd.read_excel(os.path.join(data_dir, excel_file), engine='openpyxl')
        df = df[df['Inclusion']==1.0]
    else: 
        df = pd.read_csv(os.path.join(data_dir, excel_file))
    #Inclusion
    print("--------------Inclusion--------------")
    print('Total : ', len(df))

    print("--------------fillNA--------------")
    # data = data.dropna()
    df.fillna(0.0,inplace=True)
    print(df['REAL_STONE'].value_counts())

    #Column rename
    df.rename(columns={'환자번호': 'patient_id', '검사결과': 'blood_test', 'REAL_STONE':'target' }, inplace=True)
    #column select
    columns = ['patient_id','DUCT_DILIATATION_8MM', 'DUCT_DILIATATION_10MM', 'SBP', 'DBP',
        'HR', 'RR', 'BT', 'AGE', 'GENDER','blood_test', 'PANCREATITIS','target']
    data = df[columns]
    data['patient_id'] = data['patient_id'].astype(str)

    image_list = sorted(glob(os.path.join(data_dir,"*.nii.gz")))

    def get_patient_data(image_number):
        row = df[df['patient_id'].astype(str).str.startswith(image_number)]
        return row.iloc[0, 1:].tolist() if not row.empty else None

    data_dict = {key: [] for key in ['image_path', 'DUCT_DILIATATION_8MM', 'DUCT_DILIATATION_10MM', 'SBP', 'DBP', 'HR', 'RR', 'BT', 'AGE', 'GENDER', 'blood_test', 'PANCREATITIS', 'target']}

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
    elif modality != 'mm':
        raise AssertionError("Feature engineering error")

    #Create a DataFrame from the dictionary
    train_df = pd.DataFrame(data_dict)
    
    print("--------------Scaling--------------")
    # 스케일링할 열 선택
    columns_to_scale = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'AGE', 'blood_test']
    # MinMaxScaler 객체 생성
    scaler = MinMaxScaler()
    train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])

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

    elif mode == 'test':
        print("Test set shape:", test_data.shape)
        return test_data
    
    else:
        raise ValueError("Choose mode!")
    