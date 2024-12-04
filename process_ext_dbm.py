import pandas as pd
import os

def process_train_data(dbm_file_path, target_file_path, location_codes):
    dbm_df = pd.read_csv(dbm_file_path, encoding='utf-8-sig')
    dbm_df['DateTime'] = pd.to_datetime(dbm_df['DateTime'])
    dbm_df['month'] = dbm_df['DateTime'].dt.month
    dbm_df['day'] = dbm_df['DateTime'].dt.day
    dbm_df['hour'] = dbm_df['DateTime'].dt.hour
    dbm_df['minute'] = dbm_df['DateTime'].dt.minute

    target_df = pd.read_csv(target_file_path, encoding='utf-8-sig')
    target_df['DateTime'] = pd.to_datetime(target_df['DateTime'])
    target_df['month'] = target_df['DateTime'].dt.month
    target_df['day'] = target_df['DateTime'].dt.day
    target_df['hour'] = target_df['DateTime'].dt.hour
    target_df['minute'] = target_df['DateTime'].dt.minute

    dbm_values = [None] * len(target_df)  # Initialize with None values

    for idx, row in target_df.iterrows():
        if row['LocationCode'] in location_codes:
            dbm_value = dbm_df.loc[
                (dbm_df['month'] == row['month']) &
                (dbm_df['day'] == row['day']) &
                (dbm_df['hour'] == row['hour']) &
                (dbm_df['minute'] == row['minute']),
                'dbm'
            ]
            if not dbm_value.empty:
                dbm_values[idx] = dbm_value.values[0]

    target_df['dbm'] = dbm_values
    target_df['dbm'] = pd.to_numeric(target_df['dbm'], errors='coerce')
    target_df['dbm'] = target_df['dbm'].interpolate(method='linear').round(2)
    target_df.to_csv(target_file_path, index=False, encoding='utf-8-sig')

    # 調試輸出
    print(f"Processed {len(dbm_values)} dbm values for {target_file_path}")

def process_test_data(dbm_file_paths, test_file_path, location_codes_list):
    test_df = pd.read_csv(test_file_path, encoding='utf-8-sig')
    test_df['DateTime'] = pd.to_datetime(test_df['DateTime'])
    test_df['month'] = test_df['DateTime'].dt.month
    test_df['day'] = test_df['DateTime'].dt.day
    test_df['hour'] = test_df['DateTime'].dt.hour
    test_df['minute'] = test_df['DateTime'].dt.minute

    dbm_values = [None] * len(test_df)

    # 合併處理所有來源檔
    for dbm_path, location_codes in zip(dbm_file_paths, location_codes_list):
        dbm_df = pd.read_csv(dbm_path, encoding='utf-8-sig')
        dbm_df['DateTime'] = pd.to_datetime(dbm_df['DateTime'])
        dbm_df['month'] = dbm_df['DateTime'].dt.month
        dbm_df['day'] = dbm_df['DateTime'].dt.day
        dbm_df['hour'] = dbm_df['DateTime'].dt.hour
        dbm_df['minute'] = dbm_df['DateTime'].dt.minute

        for idx, row in test_df.iterrows():
            if row['LocationCode'] in location_codes and dbm_values[idx] is None:
                dbm_value = dbm_df.loc[
                    (dbm_df['month'] == row['month']) &
                    (dbm_df['day'] == row['day']) &
                    (dbm_df['hour'] == row['hour']) &
                    (dbm_df['minute'] == row['minute']),
                    'dbm'
                ]
                if not dbm_value.empty:
                    dbm_values[idx] = dbm_value.values[0]

    test_df['dbm'] = dbm_values
    test_df['dbm'] = pd.to_numeric(test_df['dbm'], errors='coerce')
    test_df['dbm'] = test_df['dbm'].interpolate(method='linear').round(2)
    test_df.to_csv(test_file_path, index=False, encoding='utf-8-sig')

    print(f"Processed {len(dbm_values)} dbm values for {test_file_path}")
    print(f"Number of non-null dbm values: {test_df['dbm'].count()}")

# 來源路徑
dbm_file_path_1 = "data/external_data/external_dbm/dbm_L1_14.csv"
dbm_file_path_2 = "data/external_data/external_dbm/dbm_L15_16.csv"
dbm_file_path_3 = "data/external_data/external_dbm/dbm_L17.csv"
train_data_dir = "data/train_data_processed"

# 處理 LocationCode 1-14
for i in range(1, 15):
    target_file_path = os.path.join(train_data_dir, f"L{i}_Train_combined_resampled_10T_processed.csv")
    process_train_data(dbm_file_path_1, target_file_path, location_codes=range(1, 15))

# 處理 LocationCode 15-16
for i in range(15, 17):
    target_file_path = os.path.join(train_data_dir, f"L{i}_Train_combined_resampled_10T_processed.csv")
    process_train_data(dbm_file_path_2, target_file_path, location_codes=range(15, 17))

# 處理 LocationCode 17
target_file_path = os.path.join(train_data_dir, "L17_Train_combined_resampled_10T_processed.csv")
process_train_data(dbm_file_path_3, target_file_path, location_codes=[17])

# Process for test data
test_file_path = "data/processed_test_data.csv"
dbm_file_paths = [dbm_file_path_1, dbm_file_path_2, dbm_file_path_3]
location_codes_list = [range(1, 15), range(15, 17), [17]]
process_test_data(dbm_file_paths, test_file_path, location_codes_list)

print('DBM data processed and added to train data successfully.')
print('DBM data processed and added to test data successfully.')
