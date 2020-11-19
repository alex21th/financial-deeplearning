#!/usr/bin/env python

import numpy as np
import pandas as pd

import torch
from torch import nn
from types import SimpleNamespace

from time import time, strftime, gmtime

from sklearn.preprocessing import MinMaxScaler

from loader import *
from bars import *

DATASET_DIR = 'datasets_1m/'
DATASET_PREFIX = 'BTCUSD'
MODEL_DIR = 'weights/'

params = SimpleNamespace(
    intraday_freq = pd.Timedelta('00:01:00'),
    train_data = f'{DATASET_DIR}train_data.h5',
    valid_data = f'{DATASET_DIR}valid_data.h5',
    test_data = f'{DATASET_DIR}test_data.h5',
    target_variables = ['Close'],
    # target_variables = ['Open','Close'],
    predict_at_time = pd.Timedelta('00:10:00'),
    batch_size = 128,
    sequence_length = 250,
    epochs = 4,
    hidden_size = 256,
    train = True,
    num_layers = 1,
    bidirectional = False,
    lr = 0.001,
    modelname = f'{MODEL_DIR}.pt'
)

# Read data from each dataset.
df_train = pd.read_hdf(params.train_data, mode='r')
df_valid = pd.read_hdf(params.valid_data, mode='r')
df_test = pd.read_hdf(params.test_data, mode='r')

# 5 by default because we use ALL variables as features: 'High','Low','Open','Close','Volume'.
params.input_size = len(df_train.columns)
params.output_size = len(params.target_variables)

class PricesDataset(torch.utils.data.Dataset):

    def __init__(self, df: pd.DataFrame, target_vars, shift_days, scale_x=True, seq_len=1):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """
        self.shift_steps = shift_days * 60 * 24
        # self.shift_steps = 1
        df[['y_' + y for y in target_vars]] = np.log(df[target_vars].shift(-self.shift_steps) / df[target_vars])
        self.df = df[0:-self.shift_steps].values

        if scale_x:
            X = MinMaxScaler().fit_transform(self.df[:, :-len(target_vars)])
            self.features = torch.from_numpy(X).float()
        else:
            self.features = torch.from_numpy(self.df[:, :-len(target_vars)]).float()

        self.labels = torch.from_numpy(self.df[:, -len(target_vars):]).float()
        self.seq_len = seq_len

    def __len__(self):
        """return number of points in our dataset"""
        return self.features.__len__() - self.seq_len
        # return len(self.df)

    def __getitem__(self, index):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """
        X = self.features[index:index + self.seq_len]
        y = self.labels[index:index + self.seq_len]
        return X, y


class FixedDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, context_length, target_length):
        self.X = X
        self.y = y
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

def add_shifted_returns(df, target_vars, predict_at: pd.Timedelta, freq: pd.Timedelta):
    assert predict_at.seconds % freq.seconds == 0
    shift_steps = predict_at.seconds // freq.seconds
    df[['y_' + y for y in target_vars]] = np.log(df[target_vars].shift(-shift_steps) / df[target_vars])
    df = df[0:-shift_steps].values  # Pandas to NumPy array.

    # TODO
    # Scale data!

    return df

df_train = add_shifted_returns(df=df_train, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)

tr_set = FixedDataset(X = df_train[:,:params.input_size], y = df_train[:,-params.output_size:], context_length=100, target_length=10)






tr_set = FixedDataset(X = np.arange(0,20), y = np.arange(1,21), context_length=3, target_length=2)

def titem(idx):
    X, y = tr_set.__getitem__(idx)
    print(f'{X} \nShape: {X.shape}')
    print(f'{y} \nShape: {y.shape}')

tr_set = FixedDataset(X = df.values, y = df.values[:,3], context_length=100, target_length=10)
tr_generator = torch.utils.data.DataLoader(tr_set, batch_size=128)  # Converts NumPy arrays to "torch.Tensor" automatically.


X, y = next(iter(tr_generator))
X, y = X.to(device), y.to(device)

model = LSTM_Forecaster(input_size=5, hidden_size=256, output_size=params.output_size, num_layers=1, bidirectional=False).to(device)

out = model(X.float())
print(f'                    BATCH #1')
print(f'Input X shape:      {X.shape}')
print(f'Output model shape: {out.shape}')
print(f'Targets y shape:    {y.shape}')

criterion = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.MSELoss()
loss = criterion(out[:,-10:], y)
# loss = criterion(out[:,-10:], torch.reshape(y, (y.shape[0], y.shape[1], 1)))
loss

kk = 0
for i, (data, target) in enumerate(tr_generator):
    #print(f'Batch num: {i}')
    #print(f'{data} \nShape: {data.shape}')
    #print(f'{target} \nShape: {target.shape}')
    kk += 1
    if i == 7893:
        print(i)
print(kk)

training_set = PricesDataset(df=df_train, target_vars=params.target_variables, shift_days=params.days_prediction, seq_len=params.sequence_length)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=params.batch_size)

validation_set = PricesDataset(df=df_valid, target_vars=params.target_variables, shift_days=params.days_prediction, seq_len=params.sequence_length)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=params.batch_size)

test_set = PricesDataset(df=df_test, target_vars=params.target_variables, shift_days=params.days_prediction, seq_len=params.sequence_length)
test_generator = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size)


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



training_losses = []

def train(model, optimizer, criterion, train_generator, log=False):
    model.train()
    total_loss = 0
    niterations = 0
    for X, y in train_generator:
        # Get input and target sequences from batch
        # X = torch.tensor(X, dtype=torch.long, device=device)
        # y = torch.tensor(y, dtype=torch.long, device=device)

        X, y = X.to(device), y.to(device)

        # Clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls)
        model.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        # Computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        loss.backward()
        # Causes the optimizer to take a step based on the gradients of the parameters.
        optimizer.step()

        # Training statistics
        total_loss += loss.item()
        niterations += 1
        if niterations == 200 or niterations == 500 or niterations % 1000 == 0:
            print(f'Train: iteration_number={niterations}, loss={loss.item():.2f}')

        training_losses.append(loss.item())

        if log:
            print(f'Train: iteration_number={niterations}, loss={loss.item():.2f}')

    total_loss = total_loss / niterations

    return total_loss




def validate(model, criterion, valid_generator):
    model.eval()
    total_loss = 0
    niterations = 0
    with torch.no_grad():
        for X, y in valid_generator:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)

            total_loss += loss.item()
            niterations += 1

    total_loss = total_loss / niterations

    return total_loss



def get_model():
    model = LSTM_Forecaster(input_size = params.input_size,
                            hidden_size = params.hidden_size,
                            output_size = params.output_size,
                            num_layers = params.num_layers,
                            bidirectional = params.bidirectional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)
    #criterion = torch.nn.MSELoss(reduction='sum')
    criterion = torch.nn.MSELoss()
    return model, optimizer, criterion



model, optimizer, criterion = get_model()

print(model)
for name, param in model.named_parameters():
    print(f'{name:20} {param.numel()} {list(param.shape)}')
print(f'TOTAL                {sum(p.numel() for p in model.parameters())}')


if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')



epochs = params.epochs
train_loss = []
valid_loss = []
time_per_epoch = []
print(f'Training model and validation for {epochs} epochs')
start = time()

for epoch in range(1, epochs + 1):
    loss = train(model, optimizer, criterion, training_generator)
    train_loss.append(loss)
    print(f'| epoch {epoch:03d} | train loss={loss:.2f}')

    loss = validate(model, criterion, validation_generator)
    valid_loss.append(loss)
    print(f'| epoch {epoch:03d} | valid loss={loss:.2f}')

    time_per_epoch.append(time() - start)
    print(f'---------------------------------| end epoch {epoch:03d} | time {strftime("%H:%M:%S", gmtime(time()-start))} |---------------------------------')

    model_name = f'full_network_{epoch}epoch'
    torch.save(model.state_dict(), f'weights/{model_name}.pt')

end = time()
total_time = end - start
print(f'Time elapsed: {strftime("%H:%M:%S", gmtime(time()-start))}')




