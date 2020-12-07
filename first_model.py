#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

import pickle
from types import SimpleNamespace
from time import time, strftime, gmtime

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
    target_variables = ['Close'],
    predict_at_time = pd.Timedelta('00:10:00'),
    context_length = 200,
    target_length = 20,
    batch_size = 64,
    epochs = 10,
    hidden_size = 32,
    train = True,
    num_layers = 1,
    bidirectional = False,
    lr = 0.001,
    modelname = 'batch_64_hidden_32'
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

# 5 by default because we use ALL variables as features: 'High','Low','Open','Close','Volume'.
params.input_size = len(df_train.columns)
params.output_size = len(params.target_variables)

def add_shifted_returns(df, target_vars, predict_at: pd.Timedelta, freq: pd.Timedelta, scale=True):
    assert predict_at.seconds % freq.seconds == 0
    shift_steps = predict_at.seconds // freq.seconds
    df[['y_' + y for y in target_vars]] = np.log(df[target_vars].shift(-shift_steps) / df[target_vars])
    df = df[0:-shift_steps].values  # Pandas to NumPy array.

    if scale:
        # Scale X data only.
        scaler = MinMaxScaler((-1, 1))  # Default=(0, 1)
        df[:, :params.input_size] = scaler.fit_transform(df[:, :params.input_size])
        return df, scaler

    return df, None


msg('Add shifted returns and scaling data')
df_train, scaler_train = add_shifted_returns(df=df_train, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)
df_valid, scaler_valid = add_shifted_returns(df=df_valid, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)
df_test, scaler_test = add_shifted_returns(df=df_test, target_vars=params.target_variables, predict_at=params.predict_at_time, freq=params.intraday_freq)


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


msg('DATALOADER')
training_set = FixedDataset(X = df_train[:,:params.input_size], y = df_train[:,-params.output_size:], context_length=params.context_length, target_length=params.target_length)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=params.batch_size)

validation_set = FixedDataset(X = df_valid[:,:params.input_size], y = df_valid[:,-params.output_size:], context_length=params.context_length, target_length=params.target_length)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=params.batch_size)

test_set = FixedDataset(X = df_test[:,:params.input_size], y = df_test[:,-params.output_size:], context_length=params.context_length, target_length=params.target_length)
test_generator = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size)


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

        if log:
            print(f'Train: iteration_number={niterations}, accuracy={ncorrect / nelements * 100:.2f}%, loss={loss.item():.3f}')

    print(f'LA LOSS BIEN? nelements: {nelements} VS len_TR_generator: {len(train_generator)}')
    # total_loss = total_loss / niterations
    total_loss = total_loss / nelements     ## CHANGED!!!
    # total_loss = total_loss / len(train_generator)
    accuracy = ncorrect / nelements * 100

    return accuracy, total_loss


def validate(model, criterion, valid_generator):
    model.eval()
    total_loss = 0
    niterations = 0
    ncorrect = 0
    nelements = 0
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

    # total_loss = total_loss / niterations
    total_loss = total_loss / nelements  ## CHANGED!!!
    accuracy = ncorrect / nelements * 100

    return accuracy, total_loss



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



epochs = params.epochs
train_accuracy = []
train_loss = []
valid_loss = []
valid_accuracy = []
time_per_epoch = []
msg(f'Training model and validation for {epochs} epochs')
start = time()

base_loss_train = np.power(training_set.y*100, 2).mean()
print(f'BASELINE training loss (based on MSE): {base_loss_train:.2f}')
base_loss_valid = np.power(validation_set.y*100, 2).mean()
print(f'BASELINE validation loss (based on MSE): {base_loss_valid:.2f}')

for epoch in range(1, epochs + 1):
    print(f'training_generator len: {len(training_generator)}')
    acc, loss = train(model, optimizer, criterion, training_generator)
    train_accuracy.append(acc)
    train_loss.append(loss)
    print(f'| epoch {epoch:03d} | train accuracy={acc:.2f}%, train loss={loss:.6f}')

    acc, loss = validate(model, criterion, validation_generator)
    valid_accuracy.append(acc)
    valid_loss.append(loss)
    print(f'| epoch {epoch:03d} | valid accuracy={acc:.2f}%, valid loss={loss:.6f}')

    adjust_learning_rate(optimizer, epoch, 4)

    time_per_epoch.append(time() - start)
    print(f'---------------------------------| end epoch {epoch:03d} | time {strftime("%H:%M:%S", gmtime(time()-start))} |---------------------------------')

    model_name = f'{params.modelname}_{epoch}epoch'
    torch.save(model.state_dict(), f'weights/{model_name}.pt')

end = time()
total_time = end - start
print(f'Time elapsed: {strftime("%H:%M:%S", gmtime(time()-start))}')


# Save measurements.

msg('SAVE MEASURES')
measures = {}

measures['train_accuracy'] = train_accuracy
measures['training_losses'] = training_losses
measures['train_loss'] = train_loss

measures['valid_accuracy'] = valid_accuracy
measures['valid_loss'] = valid_loss

measures['time_per_epoch'] = time_per_epoch

with open(f'stats/{model_name}_stats.pickle', 'wb') as f:
    pickle.dump(measures, f)


plots_dir = 'plots/'

plt.plot(measures['train_loss'], label = 'train loss')
plt.plot(measures['valid_loss'], label = 'valid loss')
plt.legend(loc='upper right')
plt.title(f'{model_name} LOSS PER EPOCH')
plt.show()
plt.savefig(f'{plots_dir}{model_name}_LOSS_PLOT.png')
plt.clf()

plt.plot(measures['train_accuracy'], label = 'train accuracy')
plt.plot(measures['valid_accuracy'], label = 'valid accuracy')
plt.legend(loc='upper right')
plt.title(f'{model_name} ACCURACY PER EPOCH')
plt.show()
plt.savefig(f'{plots_dir}{model_name}_ACCURACY_PLOT.png')
plt.clf()

plt.plot(measures['training_losses'], label = 'train loss')
plt.legend(loc='upper right')
plt.title(f'{model_name} LOSS PER TRAINING ITERATION')
plt.show()
plt.savefig(f'{plots_dir}{model_name}_LOSS_PLOT_ITER.png')
plt.clf()





