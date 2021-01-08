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
MODEL_DIR = 'final_weights/'
STATS_DIR = 'final_stats/'
PLOTS_DIR = 'final_plots/'

params = SimpleNamespace(
    intraday_freq = pd.Timedelta('00:01:00'),
    train_data = f'{DATASET_DIR}train_data.h5',
    valid_data = f'{DATASET_DIR}valid_data.h5',
    test_data = f'{DATASET_DIR}test_data.h5',
    target_variables = ['Close'],
    technical_indicators = True,
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
parser.add_argument(f'--lr', default=params.lr, type=float)
parser.add_argument(f'--modelname', default=params.modelname, type=str)

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


def add_shifted_returns(df: pd.DataFrame, target_vars: list, predict_at: pd.Timedelta, freq: pd.Timedelta):
    assert predict_at.seconds % freq.seconds == 0
    shift_steps = predict_at.seconds // freq.seconds
    df[['y_' + y for y in target_vars]] = np.log(df[target_vars].shift(-shift_steps) / df[target_vars]) * 100
    df.dropna(inplace=True)  # We miss the first n-"predict_at" samples. (10 samples)
    return


msg('Add shifted returns')
add_shifted_returns(df=df_test, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)
print(f'Columns: {list(df_test.columns)}')

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

msg('Add technical indicators')
if params.technical_indicators:
    add_technical_indicators(df_test, window=60)
    print(f'Columns: {list(df_test.columns)}')
else:
    print('No')


def scale_data(df, X_cols):
    # Scale X data only.
    scaler = MinMaxScaler((-1, 1))  # Default=(0, 1)
    df[X_cols] = scaler.fit_transform(df[X_cols])
    return scaler


msg('Scale data')
y_columns = ['y_' + y for y in params.target_variables]
X_columns = [col for col in df_test.columns if col not in y_columns]
params.input_size = len(X_columns)
params.output_size = len(params.target_variables)

scaler_test = scale_data(df_test, X_cols=X_columns)


class FixedDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, context_length, target_length):
        self.X = X.values
        self.y = y.values
        self.context_length = context_length
        self.target_length = target_length
        self.length = (self.X.shape[0] - self.context_length) // self.target_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i0 = idx*self.target_length
        i1 = idx*self.target_length + self.context_length + self.target_length

        t0 = i0 + self.context_length
        t1 = i1

        input = self.X[i0:i1]
        target = self.y[t0:t1]

        assert target.shape[0] == self.target_length

        return input, target


msg('DATALOADER')
test_set = FixedDataset(X = df_test[X_columns], y = df_test[y_columns], context_length=params.context_length, target_length=params.target_length)
test_generator = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size)

print(f'      Test sequences: {len(test_generator.dataset):>11,}')

torch.set_default_tensor_type('torch.DoubleTensor')
# Select device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("WARNING: Training without GPU can be very slow!")
# device = torch.device('cpu')


def test(model, generator):
    model.eval()
    niterations = 0
    ncorrect = 0
    nelements = 0
    predictions = []
    with torch.no_grad():
        for X, y in generator:
            X, y = X.to(device), y.to(device)
            output = model(X)
            output_target = output[:, -params.target_length:]

            ncorrect += torch.sum(torch.sign(output_target) == torch.sign(y)).item()
            nelements += y.numel()

            niterations += 1

            predictions.append(output_target.detach().cpu().numpy())

    accuracy = ncorrect / nelements * 100

    return accuracy, predictions



class LSTM_Forecaster(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.linear(output)
        return output



def get_model():
    model = LSTM_Forecaster(input_size = params.input_size,
                            hidden_size = params.hidden_size,
                            output_size = params.output_size,
                            num_layers = params.num_layers,
                            bidirectional = params.bidirectional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)
    criterion = torch.nn.MSELoss(reduction='sum')
    # criterion = torch.nn.MSELoss()
    return model, optimizer, criterion


model, optimizer, criterion = get_model()

msg('MODEL PARAMETERS')
print(model)
for name, param in model.named_parameters():
    print(f'{name:20} {param.numel()} {list(param.shape)}')
print(f'TOTAL                {sum(p.numel() for p in model.parameters())}')



if device.type == 'cuda':
    msg('GPU INFO')
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


# Needed to compute "threshold_accuracy".
def get_Ys(generator):
    y_values = []
    for X, y in generator:
        y_values.append(y)
    y_values = np.concatenate(y_values).flatten()
    return y_values

msg("Collecting y's")
y_test = get_Ys(test_generator)

def overall_accuracy(pred, y):
    pred = np.concatenate(pred).flatten()
    acc_th = np.sum(np.sign(pred) == np.sign(y)) / len(pred) * 100
    return acc_th

def threshold_accuracy(pred, y, pct=1):
    pred = np.concatenate(pred).flatten()
    threshold = np.percentile(np.sort(np.abs(pred)), 100-pct)
    mask = np.abs(pred) >= threshold
    # print(sum(mask))
    acc_th = np.sum(np.sign(pred[mask]) == np.sign(y[mask])) / len(pred[mask]) * 100
    return acc_th


msg('Loading seed models (to ensemble)')
TRAINED_MODEL_NAME = 'b16384_e30_1epoch'  # !!!! CHANGE IF NECESSARY !!!!

seeds = (1,2,3,4,5)

all_test_accuracy = []
all_predictions = []
for s in seeds:
    model_seed_name = f's{s}_{TRAINED_MODEL_NAME}'
    model_path = f'{MODEL_DIR}{model_seed_name}.pt'
    model.load_state_dict(torch.load(model_path))
    print(model_path)
    test_accuracy, predictions = test(model, test_generator)
    all_test_accuracy.append(test_accuracy)
    all_predictions.append(predictions)

mean_test_accuracy = np.mean(all_test_accuracy)
mean_predictions = np.mean(np.array(all_predictions, dtype="object"), axis=0)

test_accuracy = overall_accuracy(mean_predictions, y_test)
test_th_accuracy = threshold_accuracy(mean_predictions, y_test, params.acc_th)

msg('RESULTS ON TEST DATA')
print(f'               Test accuracy (seed mean): {mean_test_accuracy}')
print(f'          Test accuracy (ensemble model): {test_accuracy}')
print(f'Threshold test accuracy (ensemble model): {test_th_accuracy}')



### GAINS

params = SimpleNamespace(
    intraday_freq = pd.Timedelta('00:01:00'),
    # train_date = ('2017-05', '2019-07'),
    # valid_date = ('2019-07', '2020-01'),
    # test_date  = ('2020-01', '2020-10')
    train_date = ('2017-05', '2020-01'),
    valid_date = ('2020-01', '2020-07'),
    test_date  = ('2020-07', '2020-10')
)

def info(df):
    print(f'Number of samples: {len(df):>11,}')
    print(f'Columns: {list(df.columns)}')


DATASET_PREFIX = 'BTCUSD'
dataset_dir = '../../footanalytics/projects/data/dukascopy/'

df = df_load(folder=dataset_dir, symbol=DATASET_PREFIX, start=params.test_date[0], end=params.test_date[1])
info(df)
df = get_timebars(df, time_frame=params.intraday_freq)
info(df)

def logret_buy(bars):
    return (bars[('bid','last')].apply(np.log) - bars[('ask','first')].apply(np.log))

def logret_sell(bars):
    return (bars[('bid','first')].apply(np.log) - bars[('ask','last')].apply(np.log))


r_buy = logret_buy(df)
r_sell = logret_sell(df)
comission = 2e-5
gain = np.where(df, r_buy-comission, np.where(df, r_sell-comission, 0))
