import os
import pandas as pd

def resample(df, freq='10min'):
    # 重採樣10T後的數據
    df_resampled = df.resample(freq).mean().round(2)

    return df_resampled

# 採樣10T後的數據，處理極端值
def process_to_10T(folder_path, output_folder_path, freq='10min'):
    print("處理資料採樣...")
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # 轉換 DateTime 欄位為 datetime 格式
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            
            # 移除重複的 'DateTime' 取平均值
            df = df.groupby('DateTime', as_index=False).mean()

            # 處理極端值
            df["Pressure(hpa)"] = df["Pressure(hpa)"].apply(lambda x: 1040 if x > 1040 else x)
            df["Pressure(hpa)"] = df["Pressure(hpa)"].apply(lambda x: 900 if x < 900 else x)
            df["Humidity(%)"] = df["Humidity(%)"].apply(lambda x: 100 if x > 100 else x)
            df["Temperature(°C)"] = df["Temperature(°C)"].apply(lambda x: 40 if x > 40 else x)
            df["Temperature(°C)"] = df["Temperature(°C)"].apply(lambda x: 5 if x < 5 else x)

            # 重採樣10T後的數據
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.set_index('DateTime', inplace=True)
            df = resample(df, freq=freq)
            df.reset_index(inplace=True)

            # 檢查是否有遺漏的數據
            print(f"{file_name} 檢查缺值:")
            print(df.isnull().sum())

            # 存儲處理後的數據
            os.makedirs(output_folder_path, exist_ok=True)
            output_file_path = os.path.join(output_folder_path, file_name.replace('.csv', '_resampled_10T.csv'))
            df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

def combine_train_data(output_folder_path):
    # 定義兩個資料夾路徑
    original_folder = "data/36_TrainingData/"
    additional_folder = "data/36_TrainingData_Additional_V2/"
    
    # 建立字典儲存每個地點的資料
    location_data = {}
    
    # 處理原始訓練資料
    for i in range(1, 18):
        file_name = f'L{i}_Train.csv'
        file_path = os.path.join(original_folder, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            location_data[i] = df
            print(f"已讀取原始資料: {file_name}")
    
    # 處理額外的訓練資料
    additional_locations = [2, 4, 7, 8, 9, 10, 12]
    for loc in additional_locations:
        file_name = f'L{loc}_Train_2.csv'
        file_path = os.path.join(additional_folder, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            # 合併相同地點的資料
            if loc in location_data:
                location_data[loc] = pd.concat([location_data[loc], df], ignore_index=True)
                # 排序並移除重複的資料
                location_data[loc]['DateTime'] = pd.to_datetime(location_data[loc]['DateTime'])
                location_data[loc] = location_data[loc].sort_values('DateTime')
                location_data[loc] = location_data[loc].drop_duplicates(subset=['DateTime'])
            print(f"已合併額外資料: {file_name}")
    
    # 確保輸出資料夾存在
    os.makedirs(output_folder_path, exist_ok=True)
    
    # 儲存合併後的資料
    for loc, df in location_data.items():
        output_file = os.path.join(output_folder_path, f'L{loc}_Train_combined.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"已儲存合併資料: L{loc}_Train_combined.csv")

def drop_missing_values(input_folder_path):
    for file in os.listdir(input_folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(input_folder_path, file)
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            df.dropna(inplace=True)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

            # 檢查是否有遺漏的數據列
            print(f"{file} 檢查缺值:")
            print(df.isnull().sum())
    print("已刪除缺值")

if __name__ == "__main__":    
    # 結合v1, v2訓練數據 
    combined_output_path = "data/combined_train_data"
    combine_train_data(combined_output_path)

    # 處理極端值、採樣10min的數據
    resampled_folder_path = 'data/data_resampled/'
    process_to_10T("data/combined_train_data", resampled_folder_path, freq='10min')  

    # drop 缺值
    drop_missing_values(resampled_folder_path)  