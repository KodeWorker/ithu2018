# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def root_mean_squared_error(y_ture, y_pred):
    return sqrt(mean_squared_error(y_ture, y_pred))

import matplotlib.pyplot as plt

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
    
    Xf = np.array([])
    for i in range(len(X)):
        print('\rProgress: {:.2f}%'.format((i+1)/len(X)*100), end='\r')
        
        Xftemp = np.array([])
        for j in range(n_features):
            Xtemp = X[i, :, j]
            Xfseg = np.array([])
            for k in range(int(n_timesteps/1500)):
                if len(Xfseg) == 0:
                    Xfseg = Xtemp[k*1500:(k+1)*1500]
                else:
                    Xfseg = np.vstack((Xfseg, Xtemp[k*1500:(k+1)*1500]))
            Xftemp = np.append(Xftemp, np.array([np.max(np.absolute(np.fft.fft(Xfseg[0,:]))),
                                                 np.max(np.absolute(np.fft.fft(Xfseg[3,:]))),
                                                 np.max(np.absolute(np.fft.fft(Xfseg[4,:])))]))
        
        if len(Xf) == 0:
            Xf = Xftemp
        else:
            Xf = np.vstack((Xf, Xftemp))
    print('\nDone!')
    
    # Step II: Test Data
    data_dir = os.path.join(os.path.dirname(__file__), 'sample_data', '831測驗集')
    train_filename_list = os.listdir(data_dir)
        
    X_ = []
    for train_filename in train_filename_list:
        
        data_path = os.path.join(data_dir, train_filename)
        data = pd.read_excel(data_path, header=None)
        
        # Get Features
        feature = np.array(data.iloc[:n_timesteps, :n_features].values, dtype=float)
        
        X_.append(feature)
    
    X_ = np.array(X_)
    
    X_test = np.array([])
    for i in range(len(X_)):
        print('\rProgress: {:.2f}%'.format((i+1)/len(X_)*100), end='\r')
        
        Xftemp = np.array([])
        for j in range(n_features):
            Xtemp = X_[i, :, j]
            Xfseg = np.array([])
            for k in range(int(n_timesteps/1500)):
                if len(Xfseg) == 0:
                    Xfseg = Xtemp[k*1500:(k+1)*1500]
                else:
                    Xfseg = np.vstack((Xfseg, Xtemp[k*1500:(k+1)*1500]))
            Xftemp = np.append(Xftemp, np.array([np.max(np.absolute(np.fft.fft(Xfseg[0,:]))),
                                                 np.max(np.absolute(np.fft.fft(Xfseg[3,:]))),
                                                 np.max(np.absolute(np.fft.fft(Xfseg[4,:])))]))
        
        if len(X_test) == 0:
            X_test = Xftemp
        else:
            X_test = np.vstack((X_test, Xftemp))
    print('\nDone!')
    
    n_splits = 5
    RANDOM_STATE = 777
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(Xf)
    X_test = scaler.transform(X_test)
    rf = RandomForestRegressor(n_estimators=1000, max_depth=15, random_state=RANDOM_STATE)
    rf.fit(Xf, y)
    y_rf = rf.predict(X_test)
    
    plt.figure()
    plt.plot(y_rf)