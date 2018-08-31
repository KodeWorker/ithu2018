# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt

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
    fold_y_hat = []
    fold_y_val = []    
    for train, val in kf.split(Xf):
        X_train, X_val, y_train, y_val = Xf[train], Xf[val], y[train], y[val]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Support Vector Regression
        svr = SVR(C=1.0, epsilon=1e-5)
        svr.fit(X_train, y_train)
        y_svr = svr.predict(X_val)
        loss_svr = root_mean_squared_error(y_val, y_svr)
        # Random Forests
        rf = RandomForestRegressor(n_estimators=1000, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        y_rf = rf.predict(X_val)
        loss_rf = root_mean_squared_error(y_val, y_rf)
        # Lasso Regression
        lasso = Lasso(alpha=1.0)
        lasso.fit(X_train, y_train)
        y_lasso = lasso.predict(X_val)
        loss_lasso = root_mean_squared_error(y_val, y_lasso)
        
        total_model_loss = loss_svr + loss_rf + loss_lasso
        w_svr = total_model_loss/loss_svr
        w_rf = total_model_loss/loss_rf
        w_lasso = total_model_loss/loss_lasso
        y_pred = y_svr*(w_svr/(w_svr+w_rf+w_lasso)) + y_rf*(w_rf/(w_svr+w_rf+w_lasso)) + y_lasso*(w_lasso/(w_svr+w_rf+w_lasso))
        
        fold_rmse.append(root_mean_squared_error(y_val, y_pred))
        
        fold_y_hat.append(y_pred)
        fold_y_val.append(y_val)
#        break # temp
    mean_rmse = np.mean(fold_rmse)
    
    fold_y_hat = np.array(fold_y_hat).flatten()
    fold_y_val = np.array(fold_y_val).flatten()
    plt.figure()
    plt.plot(fold_y_val, 'red', label='y_val')
    plt.plot(fold_y_hat, 'blue', label='y_hat')
    plt.legend()
    
    print('RMSE: {:.4f}'.format(mean_rmse))