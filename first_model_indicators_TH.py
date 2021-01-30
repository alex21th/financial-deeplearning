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
    lr = 0.001,
    adjust_lr = 5,
    modelname = 'batch_64_hidden_32',
    seed = 2104,
    acc_th = 1
)

custom_params = ['context_length','target_length','batch_size','epochs','hidden_size','num_layers','adjust_lr']
parser = argparse.ArgumentParser()
for p in custom_params:
    parser.add_argument(f'--{p}', default=params.__dict__[p], type=int)
args = parser.parse_args()
for p in custom_params:
    params.__dict__[p] = args.__dict__[p]

params.modelname = f'indicators_TH_c{params.context_length}_t{params.target_length}_b{params.batch_size}_h{params.hidden_size}_e{params.epochs}'

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
# params.input_size = len(df_train.columns)
# params.output_size = len(params.target_variables)

# def add_shifted_returns(df, target_vars, predict_at: pd.Timedelta, freq: pd.Timedelta, scale=True):
#     assert predict_at.seconds % freq.seconds == 0
#     shift_steps = predict_at.seconds // freq.seconds
#     df[['y_' + y for y in target_vars]] = np.log(df[target_vars].shift(-shift_steps) / df[target_vars]) * 100
#     df = df[0:-shift_steps].values  # Pandas to NumPy array.
#
#     if scale:
#         # Scale X data only.
#         scaler = MinMaxScaler((-1, 1))  # Default=(0, 1)
#         df[:, :params.input_size] = scaler.fit_transform(df[:, :params.input_size])
#         return df, scaler
#
#     return df, None
#
#
# msg('Add shifted returns and scaling data')
# df_train, scaler_train = add_shifted_returns(df=df_train, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)
# df_valid, scaler_valid = add_shifted_returns(df=df_valid, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)
# df_test, scaler_test = add_shifted_returns(df=df_test, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)


def add_shifted_returns(df: pd.DataFrame, target_vars: list, predict_at: pd.Timedelta, freq: pd.Timedelta):
    assert predict_at.seconds % freq.seconds == 0
    shift_steps = predict_at.seconds // freq.seconds
    df[['y_' + y for y in target_vars]] = np.log(df[target_vars].shift(-shift_steps) / df[target_vars]) * 100
    df.dropna(inplace=True)  # We miss the first n-"predict_at" samples. (10 samples)
    return


msg('Add shifted returns')
add_shifted_returns(df=df_train, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)
add_shifted_returns(df=df_valid, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)
add_shifted_returns(df=df_test, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)
print(f'Columns: {list(df_train.columns)}')

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
    add_technical_indicators(df_train, window=60)
    add_technical_indicators(df_valid, window=60)
    add_technical_indicators(df_test, window=60)
    print(f'Columns: {list(df_train.columns)}')
else:
    print('No')


def scale_data(df, X_cols):
    # Scale X data only.
    scaler = MinMaxScaler((-1, 1))  # Default=(0, 1)
    df[X_cols] = scaler.fit_transform(df[X_cols])
    return scaler


msg('Scale data')
y_columns = ['y_' + y for y in params.target_variables]
X_columns = [col for col in df_train.columns if col not in y_columns]
params.input_size = len(X_columns)
params.output_size = len(params.target_variables)

scaler_train = scale_data(df_train, X_cols=X_columns)
scaler_valid = scale_data(df_valid, X_cols=X_columns)
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
training_set = FixedDataset(X = df_train[X_columns], y = df_train[y_columns], context_length=params.context_length, target_length=params.target_length)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=params.batch_size)

validation_set = FixedDataset(X = df_valid[X_columns], y = df_valid[y_columns], context_length=params.context_length, target_length=params.target_length)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=params.batch_size)

test_set = FixedDataset(X = df_test[X_columns], y = df_test[y_columns], context_length=params.context_length, target_length=params.target_length)
test_generator = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size)

print(f'  Training sequences: {len(training_generator.dataset):>11,}')
print(f'Validation sequences: {len(validation_generator.dataset):>11,}')
print(f'      Test sequences: {len(test_generator.dataset):>11,}')

torch.set_default_tensor_type('torch.DoubleTensor')
# Select device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("WARNING: Training without GPU can be very slow!")
# device = torch.device('cpu')

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


def adjust_learning_rate(optimizer, epoch, n=30):
    """Sets the learning rate to the initial LR decayed by 10 every "n" epochs"""
    lr = params.lr * (0.1 ** (epoch // n))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'learning rate adjusted to: {lr:>6}')


training_losses = []

def train(model, optimizer, criterion, train_generator, log=False):
    model.train()
    total_loss = 0
    niterations = 0
    ncorrect = 0
    nelements = 0
    predictions = []
    for X, y in train_generator:

        X, y = X.to(device), y.to(device)

        # Clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls)
        model.zero_grad()
        output = model(X)
        output_target = output[:, -params.target_length:]
        loss = criterion(output_target, y)  # We only take into account the last "n-target" samples of the sequence.
        # Computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        loss.backward()
        # Causes the optimizer to take a step based on the gradients of the parameters.
        optimizer.step()

        ncorrect += torch.sum(torch.sign(output_target) == torch.sign(y)).item()
        nelements += y.numel()

        # Training statistics
        total_loss += loss.item()
        niterations += 1
        # if niterations == 200 or niterations == 500 or niterations % 1000 == 0:
        #     print(f'Train: iteration_number={niterations}, accuracy={ncorrect / nelements * 100:.2f}%, loss={loss.item():.3f}')

        training_losses.append(loss.item())

        # output_target = output_target.detach().numpy()
        predictions.append(output_target.detach().cpu().numpy())

        if log:
            print(f'Train: iteration_number={niterations}, accuracy={ncorrect / nelements * 100:.2f}%, loss={loss.item():.3f}')

    # total_loss = total_loss / niterations
    total_loss = total_loss / nelements     ## CHANGED!!!
    # total_loss = total_loss / len(train_generator)
    accuracy = ncorrect / nelements * 100

    return accuracy, total_loss, predictions


def validate(model, criterion, valid_generator):
    model.eval()
    total_loss = 0
    niterations = 0
    ncorrect = 0
    nelements = 0
    predictions = []
    with torch.no_grad():
        for X, y in valid_generator:
            X, y = X.to(device), y.to(device)
            output = model(X)
            output_target = output[:, -params.target_length:]
            loss = criterion(output_target, y)

            ncorrect += torch.sum(torch.sign(output_target) == torch.sign(y)).item()
            nelements += y.numel()

            total_loss += loss.item()
            niterations += 1

            # output_target = output_target.detach().numpy()
            predictions.append(output_target.detach().cpu().numpy())

    # total_loss = total_loss / niterations
    total_loss = total_loss / nelements  ## CHANGED!!!
    accuracy = ncorrect / nelements * 100

    return accuracy, total_loss, predictions



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

msg(params.modelname)

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
y_train = get_Ys(training_generator)
y_valid = get_Ys(validation_generator)
y_test = get_Ys(test_generator)

def threshold_accuracy(pred, y, pct=1):
    pred = np.concatenate(pred).flatten()
    threshold = np.percentile(np.sort(np.abs(pred)), 100-pct)
    mask = np.abs(pred) >= threshold
    # print(sum(mask))
    acc_th = np.sum(np.sign(pred[mask]) == np.sign(y[mask])) / len(pred[mask]) * 100
    return acc_th

epochs = params.epochs
train_accuracy = []
train_loss = []
train_th_accuracy = []
valid_accuracy = []
valid_loss = []
valid_th_accuracy = []
time_per_epoch = []
msg(f'Training model and validation for {epochs} epochs')
start = time()

base_loss_train = np.power(training_set.y, 2).mean()
print(f'BASELINE training loss (based on MSE): {base_loss_train:.2f}')
base_loss_valid = np.power(validation_set.y, 2).mean()
print(f'BASELINE validation loss (based on MSE): {base_loss_valid:.2f}')

for epoch in range(1, epochs + 1):
    acc, loss, predictions = train(model, optimizer, criterion, training_generator)
    train_accuracy.append(acc)
    train_loss.append(loss)
    acc_th = threshold_accuracy(predictions, y_train, params.acc_th)
    train_th_accuracy.append(acc_th)
    print(f'| epoch {epoch:03d} | train accuracy={acc:.2f}%, train loss={loss:.6f} | acc_threshold={acc_th:.2f}')

    acc, loss, predictions = validate(model, criterion, validation_generator)
    valid_accuracy.append(acc)
    valid_loss.append(loss)
    acc_th = threshold_accuracy(predictions, y_valid, params.acc_th)
    valid_th_accuracy.append(acc_th)
    print(f'| epoch {epoch:03d} | valid accuracy={acc:.2f}%, valid loss={loss:.6f} | acc_threshold={acc_th:.2f}')

    adjust_learning_rate(optimizer, epoch, params.adjust_lr)

    time_per_epoch.append(time() - start)
    print(f'---------------------------------| end epoch {epoch:03d} | time {strftime("%H:%M:%S", gmtime(time()-start))} |---------------------------------')

    # model_name = f'{params.modelname}_{epoch}epoch'
    # torch.save(model.state_dict(), f'weights/{model_name}.pt')

torch.save(model.state_dict(), f'weights/{params.modelname}.pt')

end = time()
total_time = end - start
print(f'Time elapsed: {strftime("%H:%M:%S", gmtime(time()-start))}')


# Save measurements.

msg('SAVE MEASURES')
measures = {}

measures['training_losses'] = training_losses

measures['train_accuracy'] = train_accuracy
measures['train_loss'] = train_loss
measures['train_th_accuracy'] = train_th_accuracy

measures['valid_accuracy'] = valid_accuracy
measures['valid_loss'] = valid_loss
measures['valid_th_accuracy'] = valid_th_accuracy

measures['time_per_epoch'] = time_per_epoch

with open(f'stats/{params.modelname}_stats.pickle', 'wb') as f:
    pickle.dump(measures, f)


plots_dir = 'plots/'

plt.plot(measures['train_loss'], label = 'train loss')
plt.plot(measures['valid_loss'], label = 'valid loss')
plt.legend(loc='upper right')
plt.title(f'{params.modelname} LOSS PER EPOCH')
plt.show()
plt.savefig(f'{plots_dir}{params.modelname}_LOSS_PLOT.png')
plt.clf()

plt.plot(measures['train_accuracy'], label = 'train accuracy')
plt.plot(measures['valid_accuracy'], label = 'valid accuracy')
plt.plot(measures['train_th_accuracy'], label = 'train th accuracy')
plt.plot(measures['valid_th_accuracy'], label = 'valid th accuracy')
plt.legend(loc='upper right')
plt.title(f'{params.modelname} ACCURACY PER EPOCH')
plt.show()
plt.savefig(f'{plots_dir}{params.modelname}_ACCURACY_PLOT.png')
plt.clf()

plt.plot(measures['training_losses'], label = 'train loss')
plt.legend(loc='upper right')
plt.title(f'{params.modelname} LOSS PER TRAINING ITERATION')
plt.show()
plt.savefig(f'{plots_dir}{params.modelname}_LOSS_PLOT_ITER.png')
plt.clf()





