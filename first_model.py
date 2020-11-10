import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from types import SimpleNamespace

from time import time, strftime, gmtime

from sklearn.preprocessing import MinMaxScaler

from loader import *
from bars import *

DATASET_DIR = 'sample_data/'
DATASET_PREFIX = 'BTCUSD'
MODEL_DIR = 'weights/'

params = SimpleNamespace(
    batch_size = 128,
    epochs = 4,
    intraday_freq = pd.Timedelta('00:05:00'),
    target_vars = ['Close'],
    train_date = ('2017-05', '2019-06'),
    valid_date = ('2019-07', '2019-12'),
    test_date  = ('2020-01', '2020-09'),
    sample_date = ('2017-05', '2017-06'),
    modelname = f'{MODEL_DIR}.pt',
    train = True,
    num_layers = 1,
    bidirectional = False,
    lr = 0.001
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


def plot_candlestick(df, N=None):
    if N is not None:
        first = np.random.randint(0, len(df)-N)
        df = df[first:first+N]
    mpf.plot(df, type='candle', style='charles', figratio=(25,8))


def plot_candlestick_steps(df, window, steps=None):
    max_steps = min(len(df), steps) if steps is not None else len(df)
    fig , ax = plt.subplots(1, max_steps-1, figsize=(6,2))
    for i in range(1, max_steps):
        c_df = df[i-1:i-1+window]
        mpf.plot(c_df, type='candle', style='charles', ax=ax[i-1])


# Load and transform dataset
print('Reading dataset...')
start = time()
df = df_load(DATASET_DIR, DATASET_PREFIX, params.sample_date[0], params.sample_date[1])
df = get_timebars(df, params.intraday_freq)
print(f'Time elapsed: {strftime("%H:%M:%S", gmtime(time()-start))}')

# Merge Ask and Bid, and format to OHLC
df = average_ask_bid(df)
df = ohlc_format(df)

class PricesDataset(torch.utils.data.Dataset):

    def __init__(self, df, target_vars, shift_days, scale_X=True):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """
        self.shift_steps = shift_days * 60 * 24
        df[['y_' + y for y in target_vars]] = np.log(df[target_vars].shift(-self.shift_steps) / df[target_vars])
        self.df = df[0:-self.shift_steps].values

        if scale_X:
            X = MinMaxScaler().fit_transform(self.df[:, :-len(target_vars)])
            self.features = torch.from_numpy(X).float()
        else:
            self.features = torch.from_numpy(self.df[:, :-len(target_vars)]).float()

        self.labels = torch.from_numpy(self.df[:, -len(target_vars):]).float()

    def __len__(self):
        """return number of points in our dataset"""
        return len(self.df)

    def __getitem__(self, idx):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """
        X = self.features[idx]
        y = self.labels[idx]
        return X, y


training_set = PricesDataset(df=df, target_vars=['Close'], shift_days=1)

training_generator = torch.utils.data.DataLoader(training_set,
                                                 batch_size=params.batch_size)


val_generator = torch.utils.data.DataLoader(x_test_scaled,
                                                 batch_size=params.batch_size)



# Select device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("WARNING: Training without GPU can be very slow!")



class LSTM_Forecaster(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
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

    total_loss = total_loss / ntokens

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

    total_loss = total_loss / ntokens

    return total_loss




params.input_size = x_train_scaled.shape[1]
params.hidden_size = 128
params.output_size = y_train_scaled.shape[1]



def get_model():
    model = LSTM_Forecaster(input_size = params.input_size,
                            hidden_size = 512,
                            #hidden_size = params.hidden_size,
                            output_size = params.output_size,
                            num_layers = 1,
                            #num_layers = params.num_layers,
                            bidirectional = params.bidirectional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)
    criterion = torch.nn.MSELoss(reduction='sum')
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
print(f'Training model-validation for {epochs} epochs')
start = time()

for epoch in range(1, epochs + 1):
    loss = train(model, optimizer, criterion, training_generator)
    train_loss.append(loss)
    print(f'| epoch {epoch:03d} | train loss={loss:.2f}')

    loss = validate(model, criterion, valid_generator)
    valid_loss.append(loss)
    print(f'| epoch {epoch:03d} | valid loss={loss:.2f}')

    time_per_epoch.append(time() - start)
    print(f'---------------------------------| epoch {epoch:03d} | time {strftime("%H:%M:%S", gmtime(time()-start))} |---------------------------------')

    model_name = f'full_network_{epoch}epoch'
    torch.save(model.state_dict(), f'weights/{model_name}.pt')

end = time()
total_time = end - start
print(f'Time elapsed: {strftime("%H:%M:%S", gmtime(time()-start))}')




