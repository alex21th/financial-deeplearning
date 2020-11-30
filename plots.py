#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from types import SimpleNamespace
from time import time, strftime, gmtime

from sklearn.preprocessing import MinMaxScaler

from loader import *
from bars import *

plt.style.use('default')


DATASET_DIR = 'datasets_60_sec/'
DATASET_PREFIX = 'BTCUSD'
MODEL_DIR = 'weights/'

params = SimpleNamespace(
    intraday_freq = pd.Timedelta('00:01:00'),
    train_data = f'{DATASET_DIR}train_data.h5',
    valid_data = f'{DATASET_DIR}valid_data.h5',
    test_data = f'{DATASET_DIR}test_data.h5',
    target_variables = ['Close'],
    predict_at_time = pd.Timedelta('00:10:00'),
    context_length = 200,
    target_length = 20,
    batch_size = 128,
    epochs = 10,
    hidden_size = 512,
    train = True,
    num_layers = 1,
    bidirectional = False,
    lr = 0.001,
    modelname = f'{MODEL_DIR}.pt'
)

def msg(text='', width=78, sym='─'):
    print('')
    if not len(text): print(sym*(width+2))
    else:
        total = width - len(text)
        left = int(np.ceil(total/2))
        right = total//2
        print(sym * left, text, sym * right)


msg('PARAMETERS')
for i in params.__dict__:
    print(f'{i:20} {params.__dict__[i]}')

def read_all_hdf(files):
    data = []
    for part, file in zip(['train', 'valid', 'test'], files):
        df = pd.read_hdf(file, mode='r')
        data.append(df)
        print(f'Number of samples ({part:>5}): {len(df):>11,}')
    print(f'Columns: {list(df.columns)}')
    return data[0], data[1], data[2]

msg(f'Load datasets from "{DATASET_DIR}"')
df_train, df_valid, df_test = read_all_hdf([params.train_data, params.valid_data, params.test_data])


def add_shifted_returns(df, target_vars, predict_at: pd.Timedelta, freq: pd.Timedelta, scale=True):
    assert predict_at.seconds % freq.seconds == 0
    shift_steps = predict_at.seconds // freq.seconds
    df[['y_' + y for y in target_vars]] = np.log(df[target_vars].shift(-shift_steps) / df[target_vars])
    df[['abs_' + y for y in target_vars]] = df[target_vars].shift(-shift_steps) - df[target_vars]
    df[['rel_' + y for y in target_vars]] = (df[target_vars].shift(-shift_steps) - df[target_vars]) / df[target_vars] * 100
    df = df[0:-shift_steps].values  # Pandas to NumPy array.

    if scale:
        # Scale X data only.
        scaler = MinMaxScaler((-1, 1))  # Default=(0, 1)
        df[:, :params.input_size] = scaler.fit_transform(df[:, :params.input_size])
        return df, scaler

    return df, None

df_train, scaler_train = add_shifted_returns(df=df_train, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq, scale=False)


plt.plot(df_train[:,3], label = 'Close price')
plt.legend(loc='upper right')
plt.title(f'Evolution close price (TRAIN DATA)')
plt.show()

plt.plot(df_train[:,6], label = 'Close price')
plt.legend(loc='upper right')
plt.title(f'Rendimientos absolutos')
plt.show()

plt.plot(df_train[:,7], label = 'Close price')
plt.legend(loc='upper right')
plt.title(f'Rendimientos relativos')
plt.show()

plt.plot(df_train[:,5], label = 'Close price')
plt.legend(loc='upper right')
plt.title(f'LOG Rendimientos relativos')
plt.show()