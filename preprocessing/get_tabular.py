import pandas as pd
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

def rename_columns(df):
    df.columns = df.columns.str.upper()
    df.rename(columns={
        'PT_NO': 'ID',
        '성별': 'SEX',
        '생년월일': 'BIRTH_DATE',
        '나이': 'AGE',
        # '검체명': 'SAMPLE',
        # '항목명': 'ITEM',
        '검사항목명': 'ITEM',
        # '검사결과': 'TEST_RESULT',
        '검사결과-수치값': 'TEST_RESULT',
        '검사시행일': 'APPOINTMENT_DATE',
        # '접수일시': 'APPOINTMENT_DATE',
        '입력일시': 'MEASUREMENT_DATE',
        # '측정일시': 'MEASUREMENT_DATE',
    }, inplace=True)
    
    return df

def calculate_age(df):
    df[['FIRST_VISIT_DATE', 'BIRTH_DATE']] = df[['FIRST_VISIT_DATE', 'BIRTH_DATE']].apply(pd.to_datetime)
    df['AGE'] = df['FIRST_VISIT_DATE'].dt.year - df['BIRTH_DATE'].dt.year - (
        (df['FIRST_VISIT_DATE'].dt.month < df['BIRTH_DATE'].dt.month) | 
        ((df['FIRST_VISIT_DATE'].dt.month == df['BIRTH_DATE'].dt.month) & 
         (df['FIRST_VISIT_DATE'].dt.day < df['BIRTH_DATE'].dt.day))
    )
    df['AGE'] = df['AGE'].astype('int64')
    
    return df.drop('BIRTH_DATE', axis=1)


def get_first_test(df):
    df = df.dropna(subset=['ITEM','TEST_RESULT'])
    # df = df.dropna(subset=['SAMPLE', 'ITEM','TEST_RESULT'])
    df['APPOINTMENT_DATE'] = pd.to_datetime(df['APPOINTMENT_DATE'])
    idx = df.groupby(['ID', 'ITEM'])['APPOINTMENT_DATE'].idxmin()
    # idx = df.groupby(['ID', 'SAMPLE', 'ITEM'])['APPOINTMENT_DATE'].idxmin()
    df = df.loc[idx].reset_index(drop=True)
    
    # df['test_type'] = df['SAMPLE'] + '_' + df['ITEM']
    df['test_type'] = df['ITEM']
    pivot_df = df.pivot_table(index='ID', columns='test_type', values='TEST_RESULT', aggfunc='first')
    pivot_df.reset_index(inplace=True)
    df = df.merge(pivot_df, on='ID')
    df = df.drop_duplicates('ID')[['ID','FIRST_VISIT_DATE','VISIBLE_STONE_CT', 'REAL_STONE','SEX','Hb', 'PLT', 'WBC', 'ALP(Alkaline phosphatase, 응급)',
        'ALT(GPT, 응급)', 'AST(GOT, 응급)', 'CRP(응급)', 'Total Bilirubin(응급)']]

    df.rename(columns={
        'ALP(Alkaline phosphatase, 응급)': 'ALP',
        'ALT(GPT, 응급)': 'ALT',
        'AST(GOT, 응급)': 'AST',
        'CRP(응급)': 'CRP',
        'Total Bilirubin(응급)': 'BILIRUBIN',
    }, inplace=True)
    df.sort_values(by='ID')
    
    return df

def get_edta_sst(df):
    # df = df[
    #     ((df['SAMPLE'].str.contains('EDTA')) & (df['ITEM'].isin(('WBC', 'Hb', 'PLT')))) |
    #     ((df['SAMPLE'].str.contains('SST')) & (df['ITEM'].isin(('ALP(Alkaline phosphatase, 응급)', 'ALT(GPT, 응급)', 'AST(GOT, 응급)', 'CRP(응급)'))))
    # ]
    df = df[
        df['ITEM'].isin(('WBC', 'Hb', 'PLT','ALP(Alkaline phosphatase, 응급)', 'ALT(GPT, 응급)', 'AST(GOT, 응급)', 'CRP(응급)', 'Total Bilirubin(응급)'))
    ]

    return df

def get_first_measurements(df):
    first_measurements = []

    for patient_id, group in df.groupby('ID'):
        group = group.sort_values(by='MEASUREMENT_DATE')
        
        for idx in range(len(group)):
            first_row = group.iloc[idx]
            try:
                first_measurement = {
                    'ID': patient_id,
                    'FIRST_MEASUREMENT_DATETIME': first_row['MEASUREMENT_DATE'],
                    'FIRST_SBP': float(first_row['SBP']),
                    'FIRST_DBP': float(first_row['DBP']),
                    'FIRST_HR': float(first_row['HR']),
                    'FIRST_RR': float(first_row['RR']),
                    'FIRST_BT': float(first_row['BT'])
                }
                first_measurements.append(first_measurement)
                break
            except ValueError:
                continue

    filtered_measurement = pd.DataFrame(first_measurements)
    
    return filtered_measurement


def merge_with_filtered_data(df, filtered_measurement):
    # df_filtered = df[['ID', 'FIRST_VISIT_DATE', 'SEX', 'AGE']]
    df_filtered = df[['ID', 'FIRST_VISIT_DATE', 'VISIBLE_STONE_CT', 'REAL_STONE', 'PANCREATITIS', 'SEX', 'AGE']]
    df_filtered = df_filtered.sort_values('FIRST_VISIT_DATE').drop_duplicates(subset=['ID'], keep='first')
    df = pd.merge(df_filtered, filtered_measurement, on='ID', how='inner')
    
    return df

def compare_his_emr(vital_his, vital_emr):
    vital_his['FIRST_VISIT_DATE'] = pd.to_datetime(vital_his['FIRST_VISIT_DATE'], errors='coerce')
    vital_emr['FIRST_VISIT_DATE'] = pd.to_datetime(vital_emr['FIRST_VISIT_DATE'], errors='coerce')
    df1_selected = vital_his[['ID', 'FIRST_VISIT_DATE']]
    df2_selected = vital_emr[['ID', 'FIRST_VISIT_DATE']]

    merged_df = pd.merge(df1_selected, df2_selected, on='ID', suffixes=('_df1', '_df2'))

    different_dates_df = merged_df[merged_df['FIRST_VISIT_DATE_df1'] != merged_df['FIRST_VISIT_DATE_df2']]

    if not different_dates_df.empty:
        print("ID with different DATA_START:")
        print(different_dates_df[['ID', 'FIRST_VISIT_DATE_df1', 'FIRST_VISIT_DATE_df2']])
    else:
        print("No IDs with different FIRST_VISIT_DATE found.")

def remove_unit(value):
    if isinstance(value, str):
        value = value.replace('×10³/㎕', '')        
        value = value.replace('mg/dL', '')
        value = value.replace('g/dL', '')
        value = value.replace('IU/L', '')
        value = value.replace('< ', '')
        try:
            return float(value)
        except ValueError:
            return value
        
    return value

def vital_preprocessing(his_df, emr_df):
        rename_columns(his_df)
        rename_columns(emr_df)
        
        # his_df = calculate_age(his_df)
        # emr_df = calculate_age(emr_df)
        
        filtered_measurement_his = get_first_measurements(his_df)
        filtered_measurement_emr = get_first_measurements(emr_df)

        
        vital_his_df = merge_with_filtered_data(his_df, filtered_measurement_his)
        vital_emr_df = merge_with_filtered_data(emr_df, filtered_measurement_emr)

        
        compare_his_emr(vital_his_df, vital_emr_df)
        
        vital_df = pd.concat([vital_his_df, vital_emr_df]).drop_duplicates()
        vital_df = vital_df.sort_values(by='ID')
        vital_df =  vital_df[['ID','SEX','FIRST_SBP','FIRST_DBP','FIRST_HR','FIRST_RR','FIRST_BT','AGE']]
        # vital_df =  vital_df[['ID','SEX','FIRST_SBP','FIRST_DBP','FIRST_HR','FIRST_RR','FIRST_BT','AGE']]
        
        return vital_df
    
def blood_preprocessing(his_df, emr_df):
        rename_columns(his_df)
        rename_columns(emr_df)
        
        result_his_df = get_edta_sst(his_df)
        result_emr_df = get_edta_sst(emr_df)
        
        compare_his_emr(result_his_df, result_emr_df)
        
        combined_df = pd.concat([result_his_df, result_emr_df]).drop_duplicates()
        filtered_df = get_first_test(combined_df)
        blood_df = filtered_df.applymap(remove_unit)
        print(blood_df.columns)
        # blood_df = blood_df[['ID','EDTA_Hb','EDTA_PLT','EDTA_WBC','SST_ALP','SST_ALT','SST_AST','SST_CRP']]
        blood_df = blood_df[['ID','Hb','PLT','WBC','ALP','ALT','AST','CRP','BILIRUBIN']]
        
        return blood_df
    
def total_preprocessing(df):
        df['SEX'] = df['SEX'].replace({'M': 1, 'F': 0})
        for col in df.columns:
            if df[col].dtype == 'int64':
                df[col].fillna(0, inplace=True)
            elif df[col].dtype == 'float64':
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
        
        return df
    
def main(data_path, vital_blood_excel, ct_excel, output):
    print('-------------------Load file-------------------')
    vital_his = pd.read_excel(os.path.join(data_path,vital_blood_excel),sheet_name=0, engine='openpyxl')
    vital_emr = pd.read_excel(os.path.join(data_path,vital_blood_excel),sheet_name=1, engine='openpyxl')
    exam_his = pd.read_excel(os.path.join(data_path,vital_blood_excel),sheet_name=2, engine='openpyxl')
    exam_emr = pd.read_excel(os.path.join(data_path,vital_blood_excel),sheet_name=3, engine='openpyxl')
    ct = pd.read_excel(os.path.join(data_path,ct_excel), engine='openpyxl')

    # Vital Sign
    print('-------------------Vital Sign preprocessing-------------------')
    vital_df = vital_preprocessing(vital_his, vital_emr)
    
    # Blood examination
    print('-------------------Blood examination preprocessing-------------------')
    blood_df = blood_preprocessing(exam_his, exam_emr)
    
    # CT examination
    print('-------------------CT examination preprocessing-------------------')
    rename_columns(ct)
    ct = ct[ct['INCLUSION']==1]
    ct.drop_duplicates(inplace=True)
    ct = ct.sort_values(by='ID')
    ct = ct[['ID','VISIBLE_STONE_CT','REAL_STONE','PANCREATITIS','DUCT_DILIATATION_10MM','DUCT_DILIATATION_8MM']]
    
    # Merge 
    print('-------------------Merge-------------------')
    subtotal_df = pd.merge(ct, blood_df, on=['ID'], how='left')
    total_df = pd.merge(vital_df, subtotal_df, on=['ID'], how='left')
    total_df = total_preprocessing(total_df)
    print(total_df.info())
    print(total_df['REAL_STONE'].value_counts())
    print(total_df.columns)
    
    # Save
    total_df.to_csv(os.path.join(data_path, output),index=False)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get json")
    parser.add_argument("--data_dir", default='/home/rkdtjdals97/datasets/DUMC_nifti/', type=str, help="data directory")
    parser.add_argument("--vital_blood_excel", default='20240927_vital.xlsx', type=str, help="vital blood excel name")
    parser.add_argument("--ct_excel", default='/home/rkdtjdals97/datasets/DUMC_nifti/bileduct_data_20241023a.xlsx', type=str, help="ct excel name")
    parser.add_argument("--output", default='dumc_1023.csv', type=str, help="output name")
    
    args = parser.parse_args()
    
    main(args.data_dir, args.vital_blood_excel, args.ct_excel, args.output)