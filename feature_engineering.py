from openfe import OpenFE, transform, tree_to_formula
import pandas as pd
import multiprocessing
from typing import Tuple
import numpy as np

def open_fe(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 實例化
    ofe = OpenFE()

    # 獲取CPU最大線程數
    n_jobs = multiprocessing.cpu_count()

    # 分離特徵
    d_x = df_train.iloc[:, df_train.columns != target]

    # 分離結果column
    d_y = df_train[[target]]

    # 訓練
    ofe.fit(data=d_x, label=d_y, n_jobs=n_jobs, 
            seed=42, 
            verbose=False, 
            feature_boosting=True,
            stage2_metric='permutation',
            task='regression',
            stage2_params={'verbose': -1,}
    )

    # 取特徵分數前5的特徵
    train_x, test_x = transform(d_x, df_test, ofe.new_features_list[:5], n_jobs=n_jobs)

    # 打印特徵
    for temp in ofe.new_features_list[:5]:
        print(tree_to_formula(temp))

    # 拼接
    train_x[target] = d_y
    return train_x.reset_index(drop=True), test_x