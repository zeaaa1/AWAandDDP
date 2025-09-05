import pywt
from tsmoothie.smoother import *
from datetime import datetime
import torch
import random
from torch.utils.data import Dataset
from pandas import read_excel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 读取数据并将 Time_x 转换为分钟数
key=2

def time_to_minutes(t):
    if isinstance(t, str):
        t = datetime.strptime(t.strip(), "%H:%M").time()
        #t = datetime.strptime(t.strip(), "%H:%M:%S").time()
    #return t.hour * 60 + t.minute * 60 + t.second
    return t.hour * 60 + t.minute

if key==1:
    data = pd.read_excel('Shandong.xlsx')
    data['Time_minute'] = data['Time'].apply(time_to_minutes)
    features_to_smooth = ['Temperature', 'Humidity', 'Wind', 'Pressure', 'Pre-Temperature']
    all_features = ['Year', 'Month', 'Date', 'Time_minute', 'Irradiance', 'Temperature', 'CloudCover', 'Wind', 'Humidity', 'Pressure', 'WindDir', 'Power', 'Pre-Irradiance', 'Pre-Temperature', 'Pre-CloudCover', 'Pre-Power']

    features_input=['Year', 'Month', 'Date', 'Time_minute', 'Irradiance', 'Temperature', 'CloudCover', 'Wind', 'Humidity', 'Pressure', 'WindDir', 'Pre-Irradiance', 'Pre-Temperature', 'Pre-CloudCover', 'Pre-Power']
    target_column = 'Power'
elif key==2:
    data = pd.read_excel('Australia.xlsx')
    data['Time_minute'] = data['Time_x'].apply(time_to_minutes)
    features_to_smooth = ['Dew Point', 'Humidity', 'Wind Speed', 'Pressure']
    all_features = ['Generator Capacity', 'Year', 'Month', 'Day', 'GG', 'Net', 'Temperature',
                    'Dew Point', 'Humidity', 'Wind Speed', 'Pressure', 'Time_minute', 'Pre-GG', 'Pre-Net',
                    'Pre-Temperature',
                    'Pre-Dew Point', 'Pre-Humidity', 'Pre-Wind Speed', 'Pre-Pressure', 'PRE-GG', 'PRE-Net',
                    'PRE-Temperature',
                    'PRE-Dew Point', 'PRE-Humidity', 'PRE-Wind Speed', 'PRE-Pressure']
    features_input = ['Generator Capacity', 'Year', 'Month', 'Day', 'Net', 'Temperature',
                      'Dew Point', 'Humidity', 'Wind Speed', 'Pressure', 'Time_minute', 'Pre-GG', 'Pre-Net',
                      'Pre-Temperature',
                      'Pre-Dew Point', 'Pre-Humidity', 'Pre-Wind Speed', 'Pre-Pressure', 'PRE-GG', 'PRE-Net',
                      'PRE-Temperature',
                      'PRE-Dew Point', 'PRE-Humidity', 'PRE-Wind Speed', 'PRE-Pressure']
    target_column = 'GG'
else:
    data = pd.read_excel('London2.xlsx')
    features_to_smooth = ['Rain', 'SolarRad']
    all_features = ['SolarEnergy', 'Month', 'Hour', 'TempOut', 'OutHum', 'DewPt', 'WindSpeed', 'WindRun', 'Bar', 'Rain', 'SolarRad', 'pre-SolarEnergy', 'pre-SolarRad']
    features_input = ['Month', 'Hour', 'TempOut', 'OutHum', 'DewPt', 'WindSpeed', 'WindRun', 'Bar', 'Rain', 'SolarRad', 'pre-SolarEnergy', 'pre-SolarRad']
    target_column = 'SolarEnergy'


scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

def replace_outliers_with_neighbors(data, features_to_check):
    for feature in features_to_check:
        if pd.api.types.is_numeric_dtype(data[feature]):
            mean = data[feature].mean()
            std = data[feature].std()
            outliers_mask = (data[feature] < (mean - std)) | (data[feature] > (mean + std))
            outlier_indices = np.where(outliers_mask)[0]

            for i in outlier_indices:
                if 0 < i < len(data) - 1:
                    data.iloc[i, data.columns.get_loc(feature)] = np.mean([
                        data.iloc[i - 1, data.columns.get_loc(feature)],
                        data.iloc[i + 1, data.columns.get_loc(feature)]
                    ])
                elif i == 0:
                    data.iloc[i, data.columns.get_loc(feature)] = data.iloc[i + 1, data.columns.get_loc(feature)]
                elif i == len(data) - 1:
                    data.iloc[i, data.columns.get_loc(feature)] = data.iloc[i - 1, data.columns.get_loc(feature)]
    return data.reset_index(drop=True)

def smooth_data(data, features_to_smooth, window_size=1):
    for feature in features_to_smooth:
        if pd.api.types.is_numeric_dtype(data[feature]):
            data[feature] = data[feature].rolling(window=window_size, min_periods=1).mean()
    return data

def inverse(y_true, y_pred):
    y_test_inversed = scaler_target.inverse_transform(y_true)
    y_pred_inversed = scaler_target.inverse_transform(y_pred)
    return y_test_inversed, y_pred_inversed

def PV_data(dataset_type='test', client=1, seq_len=48):
    PV = data.copy()

    client_ranges = {
        0: (60, 16060),
        1: (16732, 32732),
        2: (33403, 49403),
        3: (50073, 66073),
        4: (66690, 82690),
        5: (83416, 99416),
        6: (100087, 116087),
        7: (116758, 132758),
        8: (133429, 149429),
        9: (150087, 166700),
        10: (166760, 182760),
        11: (183430, 199430),
        12: (200100, 216100),
        13: (216800, 232800),
        14: (233450, 249450),
        15: (250160, 266160),
        16: (266830, 282830),
        17: (283500, 299500),
        18: (300175, 316175),
        19: (316845, 332845),
        20: (333122, 349122),
        21: (350193, 366193),
        22: (366864, 382864),
        23: (383535, 399535),
        24: (400206, 416206),
        25: (416877, 432877),
        26: (433548, 449548),
        27: (450219, 466219),
        28: (466700, 482700),
        29: (483050, 499050),
    }
    """
    client_ranges = {
        0: (2, 8502),
        1: (8750, 17250),
        2: (17550, 26050),
        3: (26300, 34800),
        4: (35000, 43500),
        5: (43710, 52210),
        6: (52500, 61000),
        7: (61200, 69700),
        8: (70000, 78500),
        9: (78700, 87200),
        10: (87500, 96000),
        11: (96200, 104700),
        12: (104900, 113400),
        13: (113700, 122200),
        14: (122500, 131000),
    }
    

    client_ranges = {
        0: (693, 8268),
        1: (18446, 26020),
        2: (32865, 38636),
        3: (45422, 52928),
        4: (60135, 67665)
    }
    """

    start_index, end_index = client_ranges.get(client, (0, len(PV)))
    PV = PV.iloc[start_index:end_index]

    PV = replace_outliers_with_neighbors(PV, all_features)
    PV = smooth_data(PV, features_to_smooth)

    features_scaled = scaler_features.fit_transform(PV[features_input])
    target_scaled = scaler_target.fit_transform(PV[[target_column]])

    X_seq, y_seq = [], []
    for i in range(seq_len, len(features_scaled)):
        X_seq.append(features_scaled[i - seq_len:i])
        y_seq.append(target_scaled[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return (X_test, y_test) if dataset_type == 'test' else (X_train, y_train)