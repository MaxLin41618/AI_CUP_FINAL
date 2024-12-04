import joblib
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import cupy as cp

def predict_power(model_path, df_test):
    # 載入測試數據
    test = df_test
    
    # 載入整個 stacking 模型
    model = joblib.load(model_path)

    # 準備特徵
    test = test.drop(['DateTime'], axis=1)
    
    # 使用完整的 stacking 模型進行預測
    predictions = model.predict(test)
    test['答案'] = predictions
    print("預測完成")
    return test

def save_submission_file(test_with_predictions, original_test_data_path, output_path):
    # 加載原始測試數據以獲取 '序號' 欄位
    original_test = pd.read_csv(original_test_data_path)
    
    # 取小數點後兩位
    test_with_predictions['答案'] = test_with_predictions['答案'].round(2)
    
    # 生成提交文件
    submission = pd.DataFrame({
        '序號': original_test['序號'],
        '答案': test_with_predictions['答案']
    })
    submission['答案'] = submission['答案'].apply(lambda x: max(0, round(x, 2)))

    submission.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"提交文件已保存到 {output_path}")


if __name__ == "__main__":
    model_path = 'models/xgboost_model.bin'
    test_data_path = 'data/processed_test_data.csv'
    original_test_data_path = 'data/upload(no answer).csv'
    submission_output_path = 'results/upload(with answer).csv'
    
    df_test = pd.read_csv(test_data_path)
    test_with_predictions = predict_power(model_path, df_test)
    save_submission_file(test_with_predictions, original_test_data_path, submission_output_path)