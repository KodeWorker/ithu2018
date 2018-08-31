# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

def root_mean_squared_error(y_ture, y_pred):
    return sqrt(mean_squared_error(y_ture, y_pred))

import matplotlib.pyplot as plt

def buildManyToOneModel(shape):
    # Reference:　https://medium.com/@daniel820710/%E5%88%A9%E7%94%A8keras%E5%BB%BA%E6%A7%8Blstm%E6%A8%A1%E5%9E%8B-%E4%BB%A5stock-prediction-%E7%82%BA%E4%BE%8B-1-67456e0a0b
    model = Sequential()
    model.add(LSTM(10, input_shape=(shape[1], shape[2])))
    # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

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
    print('\nDone!')
    
    n_splits = 5
    RANDOM_STATE = 777
    n_epochs = 100
    n_batch_size = 8
    
    scaler = StandardScaler()
    # Fit features in X_train
    for i in range(len(X)):
        scaler.partial_fit(X[i])
    # Transform features in X_train and X_val
    for i in range(len(X)):
        X[i] = scaler.transform(X[i])
    for i in range(len(X_)):
        X_[i] = scaler.transform(X_[i])
    
    model = buildManyToOneModel(X.shape)
    callback = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch_size, callbacks=[callback])

    y_lstm = model.predict(X_, batch_size=n_batch_size)
    
    plt.figure()
    plt.plot(y_lstm)