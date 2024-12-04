import os
import pandas as pd
from feature_engineering import open_fe
from testing import predict_power, save_submission_file
from training import train_model
import subprocess
import time

import warnings
warnings.filterwarnings('ignore')

def main():
    start_time = time.time()
    print("開始執行完整流程...")
    
    train_data_path = 'data/final_combined_train_data.csv'
    test_data_path = 'data/processed_test_data.csv'
    train_folder_path = 'data/train_data_processed/'
    model_path = 'models/xgboost_model.bin'
    
    # 完成所有資料前處理
    # preprocessing_scripts = ["data_cleaning.py", "preprocessing.py"]
    # for script in preprocessing_scripts:
    #     subprocess.run(["python", script])

    # 進行特徵工程
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    
    train_fe, test_fe = open_fe(df_train, df_test, 'Power(mW)')

    print(train_fe.head())
    print(test_fe.head())

    # 訓練
    train_model(train_fe, model_path)

    # 測試與生成提交文件
    test_with_predictions = predict_power(model_path, test_fe)
    original_test_data_path = 'data/upload(no answer).csv'
    submission_output_path = 'results/upload(with answer).csv'
    save_submission_file(test_with_predictions, original_test_data_path, submission_output_path)

    print(f"\n整個流程執行完成，總耗時: {(time.time() - start_time) / 60:.2f} 分鐘")


if __name__ == "__main__":
    main()