#!/usr/bin/env python

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

from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD

DATASET_DIR = 'datasets_60_sec/'
DATASET_PREFIX = 'BTCUSD'
MODEL_DIR = 'weights/'

params = SimpleNamespace(
    intraday_freq = pd.Timedelta('00:01:00'),
    train_data = f'{DATASET_DIR}train_data.h5',
    valid_data = f'{DATASET_DIR}valid_data.h5',
    test_data = f'{DATASET_DIR}test_data.h5',
    target_variables = ['Close'],
    technical_indicators = False,
    predict_at_time = pd.Timedelta('00:10:00'),
    context_length = 100,
    target_length = 10,
    batch_size = 64,
    epochs = 3,
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

custom_params = ['context_length','target_length','batch_size','epochs','hidden_size','num_layers','adjust_lr','seed']
parser = argparse.ArgumentParser()
for p in custom_params:
    parser.add_argument(f'--{p}', default=params.__dict__[p], type=int)
parser.add_argument('--lr', default=params.lr, type=float)
parser.add_argument('--modelname', default=params.modelname, type=str)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser.add_argument('--technical_indicators', default=params.technical_indicators, type=boolean_string, help='Bool type')

args = parser.parse_args()
for p in custom_params:
    params.__dict__[p] = args.__dict__[p]
params.lr = args.lr
params.modelname = args.modelname
params.technical_indicators = args.technical_indicators


# params.modelname = f'c{params.context_length}_t{params.target_length}_b{params.batch_size}_h{params.hidden_size}_e{params.epochs}'

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
