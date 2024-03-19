import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def load_data(data_dir : str, 
                mode : str = "train",
                ):
    
    print("--------------RawData--------------")
    train_df = pd.read_csv(data_dir + 'train.csv')
    print(train_df['target'].value_counts())
    
    data = train_df.loc[:, ['image_name', 'age_approx', 'anatom_site_general_challenge', 'sex', 'target']]
    data['image_name'] = f'{data_dir}/train/' + data['image_name'] + '.jpg'
    
    print("--------------DropNA--------------")
    data = data.dropna()
    print(data['target'].value_counts())
    
    print("--------------Class balance--------------")
    majority_class = data[data['target'] == 0]
    minority_class = data[data['target'] == 1]

    # Undersample the majority class to match the number of '1's in the minority class
    undersampled_majority_class = resample(majority_class,
                                        replace=False,
                                        n_samples=len(minority_class),
                                        random_state=42)

    # Concatenate minority class and undersampled majority class
    data = pd.concat([undersampled_majority_class, minority_class])
    print(data['target'].value_counts())
    
    # Define mapping dictionaries for sexes and anatom_sites
    sexes_mapping = {'male': 0, 'female': 1}
    anatom_site_mapping = {
        'torso': 0,
        'lower extremity': 1,
        'head/neck': 2,
        'upper extremity': 3,
        'palms/soles': 4,
        'oral/genital': 5,
    }
    
    # Apply mapping to the dataframe
    data['anatom_site_encoded'] = data['anatom_site_general_challenge'].map(anatom_site_mapping)
    data['sexes_encoded'] = data['sex'].map(sexes_mapping)

    # Renew data
    data = data.loc[:, ['image_name', 'age_approx', 'anatom_site_encoded', 'sexes_encoded', 'target']]

    # # Initialize StandardScaler
    # scaler = StandardScaler()
    #  # Fit the scaler to the training data and transform it
    # age_approx_train = data['age_approx'].values.reshape(-1, 1)
    # ana_approx_train = data['anatom_site_encoded'].values.reshape(-1, 1)
    
    # age_approx_train_scaled = scaler.fit_transform(age_approx_train)
    # ana_approx_train_scaled = scaler.fit_transform(ana_approx_train)
    
    # data['age_approx'] = age_approx_train_scaled.flatten()
    # data['ana_approx'] = ana_approx_train_scaled.flatten()
    
    
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
    