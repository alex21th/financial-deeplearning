#!/usr/bin/env python

import os
import sys
import errno
import shutil
import argparse
import pandas as pd

from types import SimpleNamespace
from time import time, strftime, gmtime

from loader import *
from bars import *

DATASET_PREFIX = 'BTCUSD'

params = SimpleNamespace(
    intraday_freq = pd.Timedelta('00:01:00'),
    # train_date = ('2017-05', '2019-07'),
    # valid_date = ('2019-07', '2020-01'),
    # test_date  = ('2020-01', '2020-10')
cd
    train_date = ('2017-05', '2020-01'),
    valid_date = ('2020-01', '2020-07'),
    test_date  = ('2020-07', '2020-10')
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

def info(df):
    print(f'Number of samples: {len(df):>11,}')
    print(f'Columns: {list(df.columns)}')

datasets_sizes = []

def save_data(df_name, dataset_dir, dataset_prefix, min_date, max_date, freq, save_path=''):
    msg(f'{df_name} DATASET')
    start = time()
    print('- RAW DATA')
    df = df_load(dataset_dir, dataset_prefix, min_date, max_date)
    info(df)
    df = get_timebars(df, freq)
    df = average_ask_bid(df)  # Merge Ask and Bid
    print('\n- OHLC FORMAT')
    df = ohlc_format(df)      # Format to OHLC
    info(df)
    datasets_sizes.append(len(df))
    df.to_hdf(f'{save_path}/{df_name}_data.h5', key='df', mode='w')
    print(f'Time elapsed: {strftime("%H:%M:%S", gmtime(time() - start))}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='../../footanalytics/projects/data/dukascopy/', help='Path of original data.', type=str)
    parser.add_argument('--save', default='./', help='Path where new datasets will be saved.', type=str)
    parser.add_argument('--overwrite', default=False, help='Overwrite last dataset folder created.', type=bool)

    args = parser.parse_args()

    folder_name = 'datasets_' + str(params.intraday_freq.seconds) + '_sec'
    save_directory = f'{args.save}{folder_name}'

    try:
        os.mkdir(save_directory)
    except OSError as exc:
        if not args.overwrite: raise
        else:
            shutil.rmtree(save_directory)
            os.mkdir(save_directory)
            print(f'\n    WARNING. "{save_directory}" has been overwritten!    ')


    save_data('train', args.dataset, DATASET_PREFIX, params.train_date[0], params.train_date[1], params.intraday_freq, save_directory)
    save_data('valid', args.dataset, DATASET_PREFIX, params.valid_date[0], params.valid_date[1], params.intraday_freq, save_directory)
    save_data('test', args.dataset, DATASET_PREFIX, params.test_date[0], params.test_date[1], params.intraday_freq, save_directory)

    msg('SUMMARY')
    for set, size in zip(['train', 'valid', 'test'], datasets_sizes):
        print(f'Number of samples ({set:>5}): {size:>11,} {np.round(size / np.sum(np.array(datasets_sizes)) * 100, 2):>8} %')

