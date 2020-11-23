#!/usr/bin/env python

import os
import pandas as pd
import sys

from types import SimpleNamespace
from time import time, strftime, gmtime

from loader import *
from bars import *

DATASET_PREFIX = 'BTCUSD'

params = SimpleNamespace(
    intraday_freq = pd.Timedelta('00:01:00'),
    train_date = ('2017-05', '2019-07'),
    valid_date = ('2019-07', '2020-01'),
    test_date  = ('2020-01', '2020-10')
)

def average_ask_bid(df):
    d = {'avg_price': (df.ask + df.bid) / 2, 'avg_volume': (df.ask_volume + df.bid_volume) / 2}
    avg = pd.concat(d.values(), axis=1, keys=d.keys())
    df = pd.concat([df, avg], axis=1)
    return df

def ohlc_format(df):
    df = df[['avg_price','avg_volume']]
    df.columns = df.columns.droplevel()
    df = df.rename(columns={'max': 'High', 'min': 'Low', 'first': 'Open', 'last': 'Close', 'sum': 'Volume'})
    return df

def save_data(df_name, dataset_dir, dataset_prefix, min_date, max_date, freq, save_path=''):
    print(f'Reading {df_name} dataset...')
    start = time()
    df = df_load(dataset_dir, dataset_prefix, min_date, max_date)
    df = get_timebars(df, freq)
    df = average_ask_bid(df)  # Merge Ask and Bid
    df = ohlc_format(df)      # Format to OHLC
    df.to_hdf(f'{save_path}/{df_name}_data.h5', key='df', mode='w')
    print(f'Number of {df_name} samples: {df.shape[0]}')
    print(f'Time elapsed: {strftime("%H:%M:%S", gmtime(time() - start))}')

if __name__ == "__main__":
    DATASET_DIR = sys.argv[1]
    SAVE_DATA = sys.argv[2]

    save_data('train', DATASET_DIR, DATASET_PREFIX, params.train_date[0], params.train_date[1], params.intraday_freq, SAVE_DATA)
    save_data('valid', DATASET_DIR, DATASET_PREFIX, params.valid_date[0], params.valid_date[1], params.intraday_freq, SAVE_DATA)
    save_data('test', DATASET_DIR, DATASET_PREFIX, params.test_date[0], params.test_date[1], params.intraday_freq, SAVE_DATA)

