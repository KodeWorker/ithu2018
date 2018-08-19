# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def root_mean_squared_error(y_ture, y_pred):
    return sqrt(mean_squared_error(y_ture, y_pred))

if __name__ == '__main__':
    
    # Step I: Read Data
    data_dir = os.path.join(os.path.dirname(__file__), 'sample_data', '806初賽訓練數據')
    train_filename_list = os.listdir(data_dir)
    
    n_timesteps = 7500
    n_features = 4
    
    X = []
    y = []
    for train_filename in train_filename_list:
        
        data_path = os.path.join(data_dir, train_filename)
        data = pd.read_excel(data_path, header=None)
        
        # Get Features
        feature = np.array(data.iloc[:n_timesteps, :n_features].values, dtype=float)
        # Get Target Value
        target = float(data.iloc[n_timesteps, 0].replace('加工品質量測結果:', ''))
        
        X.append(feature)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    Xf = X.reshape((len(X), n_timesteps*n_features))
    
    n_splits = 5
    RANDOM_STATE = 777
    
    fold_rmse = []
    kf = KFold(n_splits=n_splits, random_state=RANDOM_STATE)
    for train, test in kf.split(Xf):
        X_train, X_test, y_train, y_test = Xf[train], Xf[test], y[train], y[test]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        svr = SVR(C=1.0, epsilon=1e-5)
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        
        fold_rmse.append(root_mean_squared_error(y_test, y_pred))
        break # temp
    mean_rmse = np.mean(fold_rmse)
    
    print('RMSE: {:.4f}'.format(mean_rmse))