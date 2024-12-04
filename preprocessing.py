import os
import pandas as pd
import subprocess


def create_time_features(df):
    '''
    透過DateTime欄位創建時間特徵
    '''
    # df['quarter'] = df['DateTime'].dt.quarter
    df['month'] = df['DateTime'].dt.month
    df['day'] = df['DateTime'].dt.day
    df['hour'] = df['DateTime'].dt.hour
    df['minute'] = df['DateTime'].dt.minute
    return df

def process_test_data(test_data_path, output_path):
    '''
    處理測試數據，並創建時間特徵
    '''
    test = pd.read_csv(test_data_path)
    
    # Convert '序號' to string type
    test['序號'] = test['序號'].astype(str)
    
    # Extract DateTime and LocationCode from '序號'
    test['DateTime'] = pd.to_datetime(test['序號'].str[:12], format='%Y%m%d%H%M')
    test['LocationCode'] = test['序號'].str[-2:].astype(str)

    # Create time features
    test = create_time_features(test)

    # Drop column
    test.drop(['序號', '答案'], axis=1, inplace=True)
    
    # Save the processed test data
    test.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"處理測試資料後存到 {output_path}")

def process_train_data_time_feature(train_folder, output_folder_path):
    '''
    處理訓練數據，並創建時間特徵
    '''
    os.makedirs(output_folder_path, exist_ok=True)

    for file_name in os.listdir(train_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(train_folder, file_name)
            
            try:
                # 加載數據
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                
                # 轉換 'DateTime' 欄位為 datetime 格式
                df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
                
                # 移除無效的 DateTime
                df = df.dropna(subset=['DateTime'])
                
                # 創建時間特徵
                df = create_time_features(df)
                
                # 生成新的檔案名稱
                output_file_name = file_name.replace('.csv', '_processed.csv')
                output_file_path = os.path.join(output_folder_path, output_file_name)
                
                # 保存更新後的 DataFrame 到新的 CSV 檔案
                df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
                print(f"已成功處理並保存檔案：{output_file_path}")
            
            except Exception as e:
                print(f"處理檔案 {file_name} 時發生錯誤：{e}")

def load_and_combine_data(folder_path, output_path='data/combined_train_data.csv'):
    '''
    合併訓練數據
    '''
    # 合併訓練數據
    combined_df = pd.DataFrame()

    # 讀取L1到L17的檔案
    for i in range(1, 18):
        file_name = f'L{i}_Train_combined_resampled_10T_processed.csv'
        file_path = os.path.join(folder_path, file_name)
        
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                print(f"正在讀取: {file_name}")
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            else:
                print(f"警告: 找不到檔案 {file_name}")
        except Exception as e:
            print(f"處理 {file_name} 時發生錯誤: {e}")

    # 轉換 'DateTime' 欄位為 datetime 格式
    try:
        combined_df['DateTime'] = pd.to_datetime(combined_df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    except ValueError as e:
        print(f"日期時間格式錯誤: {e}")
        combined_df['DateTime'] = pd.to_datetime(combined_df['DateTime'], errors='coerce')

    # 保留特定特徵
    features = ['LocationCode', 'month', 'day', 'hour', 'minute', 'Power(mW)','GloblRad_daily', 'GloblRad_hourly', 'dbm', '仰角']
    combined_df = combined_df[features]
    
    # 刪除 hour 為 0 的極端樣本
    combined_df = combined_df[combined_df['hour'] != 0]
    
    # 刪除 'GloblRad_daily' 為缺值的樣本
    combined_df = combined_df.dropna(subset=['GloblRad_daily'])

    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"已成功合併並保存檔案：{output_path}")

    return combined_df

def add_features():
    '''
    執行其他Python檔案以新增特徵
    '''
    scripts = [
        "process_ext_daily_traindata.py",
        "process_ext_daily_testdata.py",
        "process_ext_hour_traindata.py",
        "process_ext_hour_testdata.py",
        "process_ext_dbm.py",
        "process_ext_solar_angles.py",
    ]
    
    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        try:
            subprocess.run(["python", script_path], check=True)
            print(f"成功執行完成 {script}")
        except subprocess.CalledProcessError as e:
            print(f"執行 {script} 時發生錯誤: {e}")

if __name__ == "__main__":
    train_data_folder = 'data/data_resampled'
    train_data_process_output_folder = 'data/train_data_processed/'
    test_data_path = 'data/upload(no answer).csv'
    test_data_process_output_path = 'data/processed_test_data.csv'

    process_train_data_time_feature(train_data_folder, train_data_process_output_folder)  # 處理訓練數據，並創建時間特徵

    process_test_data(test_data_path, test_data_process_output_path)  # 處理測試數據，並創建時間特徵

    add_features()  # 新增外部資料特徵

    load_and_combine_data(train_data_process_output_folder, output_path='data/final_combined_train_data.csv')  # 合併所有地點訓練數據
