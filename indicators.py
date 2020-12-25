
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

import pickle
from types import SimpleNamespace
from time import time, strftime, gmtime
import random
import argparse

from sklearn.preprocessing import MinMaxScaler

from loader import *
from bars import *

DATASET_DIR = 'datasets_60_sec/'
DATASET_PREFIX = 'BTCUSD'
MODEL_DIR = 'weights/'

params = SimpleNamespace(
    intraday_freq = pd.Timedelta('00:01:00'),
    train_data = f'{DATASET_DIR}train_data.h5',
    valid_data = f'{DATASET_DIR}valid_data.h5',
    test_data = f'{DATASET_DIR}test_data.h5',
    context_variables = ['High','Low','Open','Close','Volume'],
    target_variables = ['Close'],
    predict_at_time = pd.Timedelta('00:10:00'),
    context_length = 100,
    target_length = 10,
    batch_size = 64,
    epochs = 10,
    hidden_size = 32,
    train = True,
    num_layers = 1,
    bidirectional = False,
    lr = 0.0001,  # 0.001
    adjust_lr = 5,
    modelname = 'batch_64_hidden_32',
    seed = 2104,
    acc_th = 1
)

params.modelname = f'lr0.0001_TH_c{params.context_length}_t{params.target_length}_b{params.batch_size}_h{params.hidden_size}_e{params.epochs}'

# Fix random seed for reproducibility
def seed_everything(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(params.seed)

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

# 5 by default because we use ALL variables as features: 'High','Low','Open','Close','Volume'.
params.input_size = len(df_train.columns)
params.output_size = len(params.target_variables)

def add_shifted_returns(df, target_vars, predict_at: pd.Timedelta, freq: pd.Timedelta):
    assert predict_at.seconds % freq.seconds == 0
    shift_steps = predict_at.seconds // freq.seconds
    df[['y_' + y for y in target_vars]] = np.log(df[target_vars].shift(-shift_steps) / df[target_vars]) * 100
    # df = df[0:-shift_steps].values  # Pandas to NumPy array.
    df = df[0:-shift_steps]  # Pandas to NumPy array.
    return



msg('Add shifted returns and scaling data')
add_shifted_returns(df=df_train, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)
# df_valid, scaler_valid = add_shifted_returns(df=df_valid, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)
# df_test, scaler_test = add_shifted_returns(df=df_test, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)


# scaler = MinMaxScaler((-1, 1))  # Default=(0, 1)

# df_train[[*params.context_variables]] = scaler.fit_transform(df_train[[*params.context_variables]])

# INDICATORS PART
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD

def add_technical_indicators(df, window):
    indicator_bb = BollingerBands(close=df['Close'], n=window, ndev=2)
    indicator_rsi = RSIIndicator(close=df['Close'], n=window)
    indicator_macd = MACD(close=df['Close'], n_slow=window, n_fast=window//4, n_sign=window//11)
    indicator_stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], n=window, d_n=window//11)

    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['rsi'] = indicator_rsi.rsi()
    df['macd'] = indicator_macd.macd()
    df['stoch'] = indicator_stoch.stoch()

    df.dropna(inplace=True)  # We miss the first n-"window" samples.
    return


def scale_data(df, not_scale):
    # Scale X data only.
    X_columns = [col for col in df.columns if col not in not_scale]
    print(X_columns)
    scaler = MinMaxScaler((-1, 1))  # Default=(0, 1)
    df[X_columns] = scaler.fit_transform(df[X_columns])
    return scaler



add_technical_indicators(df_train, window=60)


scaler_train = scale_data(df_train, not_scale=['y_Close'])
