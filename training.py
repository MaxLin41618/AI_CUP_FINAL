from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import joblib
import optuna
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
import cupy as cp
import time  

import matplotlib.font_manager as fm
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'sans-serif']

def objective(trial, X, y):
    # 分割訓練集和驗證集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 將數據移至 GPU
    X_train = cp.array(X_train)
    X_val = cp.array(X_val)
    y_train = cp.array(y_train)
    y_val = cp.array(y_val)
    X = cp.array(X)
    y = cp.array(y)

    # 定義超參數搜索空間
    param = {
        'device': 'cuda:0',
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 100, 100000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }

    # 交叉驗證評估
    model = xgb.XGBRegressor(**param)
    scores = cross_val_score(model, X.get(), y.get(), cv=5, scoring='neg_mean_absolute_error')
    return -scores.mean()  # 最小化 MAE

def train_model(df_train, model_output_path):
    # 準備訓練數據
    X = df_train.drop(['Power(mW)'], axis=1)
    y = df_train['Power(mW)']
    
    # 創建 Optuna study
    # study = optuna.create_study(direction='minimize')
    # study.optimize(lambda trial: objective(trial, X, y), n_trials=20)

    # 輸出最佳參數
    # print('最佳參數:')
    # for key, value in study.best_params.items():
    #     print(f'{key}: {value}')
    
    # # 使用最佳參數訓練最終模型
    # best_params = study.best_params
    # best_params['eval_metric'] = ['mae', 'rmse']  # 擴充評估指標
    
    # 將數據分為訓練集、驗證集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 基礎模型
    base_models = [
        ('lgb', LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.1,
            max_depth=12,
            random_state=44,
            verbose=1000,          
            verbosity=-1,
            metric=['mae','rmse'], 
        )),
        ('cat', CatBoostRegressor(
            iterations=3000,
            learning_rate=0.1,
            depth=12,
            random_state=45,
            verbose=1000,
            thread_count=-1,
            loss_function='RMSE',
        )),
    ]
    
    # 最終模型
    final_estimator = XGBRegressor(
        n_estimators=3000,      
        learning_rate=0.1,
        max_depth=12,
        random_state=46,
        eval_metric=['mae', 'rmse'],
        verbosity=2,
        device='cuda',
    )
    
    model = Pipeline([
        ('stacking', StackingRegressor(
            estimators=base_models,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1,  
            passthrough=True,
            verbose=2,
        ))
    ])
    
    # 訓練模型
    print("開始訓練模型...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"訓練完成，總耗時: {elapsed_time / 60:.2f} 分鐘")

    # 驗證模型
    y_pred = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = root_mean_squared_error(y_val, y_pred)
    print(f"驗證 MAE: {mae:.2f}")
    print(f"驗證 RMSE: {rmse:.2f}")

    # 獲取原始特徵名稱
    feature_names = X_train.columns.tolist()
    
    # 如果使用了 passthrough=True，StackingRegressor 會將原始特徵與基礎模型的預測連接起來
    # 因此，需要擴展特徵名稱以匹配最終模型的輸入
    n_base_models = len(base_models)
    for i in range(n_base_models):
        feature_names.append(f'base_model_{i}_prediction')

    # 提取最終的 XGBoost 模型
    xgb_model = model.named_steps['stacking'].final_estimator_
    
    # 設置 XGBoost 模型的特徵名稱
    xgb_model.get_booster().feature_names = feature_names
    
    # 繪製特徵重要性
    plot_importance(xgb_model)  
    plt.show()

    # 保存模型
    joblib.dump(model, model_output_path)
    print(f"模型已保存到 {model_output_path}")


if __name__ == "__main__":
    train_data_path = 'data/final_combined_train_data.csv'
    model_output_path = 'models/xgboost_model.bin'

    # 訓練模型，注意不包含特徵工程的新特徵，效果較差。請到main.py中執行特徵工程
    df_train = pd.read_csv(train_data_path)
    train_model(df_train, model_output_path)
