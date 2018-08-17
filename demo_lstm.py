# -*- coding: utf-8 -*-

import os
import xlrd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

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
    
    data_dir = os.path.join(os.path.dirname(__file__), 'sample_data', '806初賽訓練數據')
    train_filename_list = os.listdir(data_dir)

    # Step I: Read Data
    n_samples = 7500
    feature_modifier = 1e6
    
    X = []
    y = []
    for train_filename in train_filename_list:
        
        data_path = os.path.join(data_dir, train_filename)
        data = xlrd.open_workbook(data_path)
        table = data.sheets()[0] 
        
        # Get Features
        # Reference:　https://stackoverflow.com/questions/27893110/python-convert-xlrd-sheet-to-numpy-matrix-ndarray
        feature = np.empty([n_samples, table.ncols], dtype=float)
        curr_row = 0
        while curr_row < n_samples: # for each row
            row = table.row(curr_row)
            if curr_row > 0: # don't want the first row because those are labels
                for col_ind, el in enumerate(row):
                    feature[curr_row - 1, col_ind] = el.value*feature_modifier
            curr_row += 1
        
        # Get Target Value
        target = float(table.cell(n_samples, 0).value.replace('加工品質量測結果:', ''))
        
        X.append(feature)
        y.append(target)
    
    # Step II: Data Quality Evaluation
    # Check for NaN value
    for i in range(len(X)):
        if np.isnan(X[i]).any():
            print('{}th data in X contains NaN'.format(i+1))
            # replace NaN with 0 (each feature ends with 0s)
            X[i] = np.nan_to_num(X[i])
    
    # Outliers - impute 0s (other methods may apllied)
    upper_bound = 1e3
    lower_bound = -1e3
    for i in range(len(X)):
        X[i][np.where(X[i] > upper_bound)] = 0
        X[i][np.where(X[i] < lower_bound)] = 0

    # Step III: 5-fold CV and Normalization
    
    n_fold = 5
    RANDOM_STATE = 777
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/n_fold, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    # Fit features in X_train
    for i in range(len(X_train)):
        scaler.partial_fit(X_train[i])
    # Transform features in X_train and X_val
    for i in range(len(X_train)):
        X_train[i] = scaler.transform(X_train[i])
    for i in range(len(X_val)):
        X_val[i] = scaler.transform(X_val[i])
    
    # Step IV: Train LSTM model
    n_epochs = 100
    n_batch_size = 16
    
    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1,1)
    X_val = np.array(X_val)
    y_val = np.array(y_val).reshape(-1,1)
    
    model = buildManyToOneModel(X_train.shape)
    callback = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size, validation_data=(X_val, y_val), callbacks=[callback])

    # Step V: Model Evaluation
    
    y_hat = model.predict(X_val, batch_size=None, verbose=0, steps=None)
    mse = mean_squared_error(y_val, y_hat)
    print('MSE: {:.4f}'.format(mse))