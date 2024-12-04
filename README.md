# AI_CUP_FINAL
## 環境配置
作業系統: Windows 11 家用版 10.0.22631  
CPU: 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz  
GPU: NVIDIA GeForce RTX 3060 12GB  
RAM: 16GB  
語言: Python 3.9.20  
主要套件(函式庫):  
lightgbm: 4.5.0  
catboost: 1.2.7  
xgboost: 2.1.1  
openfe: 0.0.12  
numpy: 1.26.4  
pandas: 2.2.2  
scikit-learn: 1.5.2  

## 程式檔  
`data_cleaning.py` 重新採樣10分鐘、處理缺值  
`preprocessing.py` 處理時間特徵與外部特徵，並整合17個地點  
`feature_engineering.py` 處理OpenFE特徵工程  
`training.py` 訓練  
`testing.py` 測試並生成提交文件  
`main.py` 一鍵完成所有流程  

`process_ext_daily_testdata.py` 新增每日日射量到測試資料  
`process_ext_daily_traindata.py` 新增每日日射量到訓練資料  
`process_ext_dbm.py` 新增每十分鐘雷達功率到訓練與測試資料  
`process_ext_hour_testdata.py` 新增每小時日射量到測試資料  
`process_ext_hour_traindata.py` 新增每小時日射量到訓練資料  
`process_ext_solar_angles.py` 新增每十分鐘仰角到訓練與測試資料  

## 資料檔
`combined_train_data`資料夾: 內容是結合v1, v2的原始資料  
`data_resampled` 資料夾: 內容為採樣10分鐘的數據  
`external_data` 資料夾: 內含所有外部資料  
`train_data_processed` 資料夾: 內含新增額外特徵後的訓練數據  
`final_combined_train_data.csv`: 整合17個地點的最終訓練資料  
`processed_test_data.csv`: 最終測試資料  
`upload(with answer).csv`: 提交的答案文件  

## 流程
1. data_cleaning  
2. preprocessing  
3. feature_engineering  
4. training  
5. testing  

## 說明
因為模型檔案約500MB，所以github內容不含模型，可以執行main.py訓練時間約1小時，或**到GoogleDrive下載zip檔**  
[This link](https://drive.google.com/file/d/1FW_ih8N6H3UndVsBThtgC2o1BERojMVB/view?usp=drive_link)  
* main.py包含執行特徵工程，可以一鍵於Terminal完成所有流程。  
* 如果只執行training.py並未包含特徵工程，效果較差。  
* public分數最佳，由於training.py中的模型參數設定n_estimators=3000、iterations=3000所以**訓練長達約1小時**   
* private分數最佳，為參數n_estimators=8000、iterations=8000，**訓練長達6小時**  