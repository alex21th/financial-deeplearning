#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from types import SimpleNamespace
from time import time
import swifter

from data_classes import Dictionary, SegmentationDataset, PadSequence


# ```
# import data_classes
# import importlib
# importlib.reload(data_classes)
# ```

# In[2]:


params = SimpleNamespace(
    train_file = '../data-humans/sample_train.csv',
    valid_file = '../data-humans/sample_valid.csv',
    predictor_variables = ['latitude', 'longitude', 'diff', 'duration', 'hour', 'week', 'month', 'local'],
    categorical_variables = ['hour', 'week', 'month', 'local'], # Modify hour embedding to sinusoidal?
    embedding_dim = 32,
    target_variables = ['local', 'diff'],
    batch_size = 512,
    max_length = 64,
    hidden_size = 256,
    epochs = 4,
    bidirectional = False,
    lr = 0.001           #Default lr for Adam optimizer is 0.001 https://pytorch.org/docs/stable/optim.html
)

params.input_size = len(params.predictor_variables) + len(params.categorical_variables) * (params.embedding_dim - 1)


# ***

# In[3]:


start_time = time()

nrows = None
df = pd.read_csv(params.train_file, nrows=nrows)
df_valid = pd.read_csv(params.valid_file, nrows=nrows*0.1 if nrows is not None else None)

dictionaries = {} 
for column in params.categorical_variables:
    print(column)
    dictionaries[column] = Dictionary(df[column].unique())
    df[column] = df[column].swifter.apply(lambda value: dictionaries[column].token2idx[value])
    df_valid[column] = df_valid[column].swifter.apply(lambda value: dictionaries[column].token2idx[value])

params.embedding_input = [len(dictionaries[column]) for column in reversed(params.categorical_variables)]
params.output_size = len(dictionaries['local']) + 1

print(time() - start_time)


# In[4]:


# df.head()


# ***

# In[5]:


training_set = SegmentationDataset(df=df,
                                   predictor_variables=params.predictor_variables,
                                   target_variables=params.target_variables,
                                   max_length=params.max_length)


# In[6]:


# Queda per veure com fer batches de longituds d'itineraris semblants per així optimitzar GPU.
training_generator = torch.utils.data.DataLoader(training_set, 
                                                 batch_size=params.batch_size,
                                                 shuffle=True,
                                                 num_workers=4,
                                                 drop_last=True,
                                                 collate_fn=PadSequence())


# In[7]:


valid_set = SegmentationDataset(df=df_valid,
                                   predictor_variables=params.predictor_variables,
                                   target_variables=params.target_variables,
                                   max_length=params.max_length)


# In[8]:


valid_generator = torch.utils.data.DataLoader(valid_set, 
                                                 batch_size=params.batch_size,
                                                 shuffle=True,
                                                 num_workers=4,
                                                 collate_fn=PadSequence())


# In[9]:


t_0 = time()
my_iter = iter(training_generator)
X, lengths, y, hashes = next(my_iter)
print(f'-------------------------------------{time() - t_0:.2f}')
t_0 = time()
X, lengths, y, hashes = next(my_iter)
print(f'-------------------------------------{time() - t_0:.2f}')


# In[10]:


print(X.shape)


# In[11]:


print(X[0].shape)


# ***

# **DNN Model**
# 
# B = Number of itineraries (batch size)
# 
# T = Max number of sensor detections in an itinerary
# 
# C = Number of columns
# 
# I = Input size (number of columns + embeddings)
# 
# O = Output size
# 
# H = Hidden size

# ***

# In[12]:


# Select device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("WARNING: Training without GPU can be very slow!")


# In[13]:


class ItineraryRNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, embedding_input, embedding_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, embedding_dim, padding_idx=0) for size in embedding_input])
        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.linear1 = torch.nn.Linear(hidden_size, output_size // 2)
        self.ln_linear1 = nn.LayerNorm(output_size // 2)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(output_size // 2, output_size)
        
    def forward(self, x, x_lengths):
        '''Input tensor x has shape T x B x C.
           The variables requiring an embedding are placed in the last positions of dim=2.
           For instance, the size of the dictionary of embeddings of x[:, :, -1]
           is given by embedding_input(0).'''
        # T x B x C
        x_embedded = x[:, :, :-len(self.embeddings)]
        for idx, embedding in enumerate(self.embeddings):
            current_embedding = embedding(x[:, :, -1 - idx].long())
            x_embedded = torch.cat([x_embedded, current_embedding.float()], dim=2)
        # T x B x I
        packed = torch.nn.utils.rnn.pack_padded_sequence(x_embedded, x_lengths, batch_first=False, enforce_sorted=False)
        # Packed T x B x I
        output, _ = self.rnn(packed)
        # Packed T x B x H
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=False, padding_value=float(0))
        # T x B x H
        y = F.relu(self.linear1(padded)) # self.bn1
        y = self.dropout(self.ln_linear1(y))
        # T x B x O/2
        output = self.linear2(y)
        # T x B x O
        return output


# In[14]:


training_losses = []
def train(model, optimizer, criterion, train_generator, log=True):
    model.train()
    total_loss = 0
    niterations = 0
    for X, lengths, y, hashes in train_generator:
        # training_lossesprint(X.shape)
        X, lengths, y = X.to(device), lengths.to(device), y.to(device)
        output = model(X, lengths)

        model.zero_grad()
        output = model(X, lengths)
        loss = criterion(output, y, lengths)
        loss.backward()
        optimizer.step()
        # print(output[:5,:])
        # Training statistics
        
        total_loss += loss.item()
        niterations += 1
        indicator = total_loss / niterations
        training_losses.append(loss.item())
        if log:
            print(f'Train: iteration_number={niterations}, loss={loss.item():.2f}')
    return indicator


# In[15]:


def validate(model, criterion, valid_generator):
    model.eval()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
    niterations = 0
    with torch.no_grad():
        for X, lengths, y, hashes in valid_generator:
            X, lengths, y = X.to(device), lengths.to(device), y.to(device)

            model.zero_grad()
            output = model(X, lengths)
            loss = criterion(output, y, lengths)
            
            total_loss += loss.item()
            niterations += 1
    
    indicator = total_loss / niterations
    return indicator


# In[16]:


def maskedLoss(prediction, target, lengths):
    nll, mse = 0, 0
    for i, length in enumerate(lengths):
        nll += F.cross_entropy(prediction[:length, i, :-1], target[:length, i, 0].long())
        mse += F.mse_loss(prediction[:length, i, -1], target[:length, i, 1])
    overall_loss = (600 * nll + 2800 * mse) / (600 + 2800)
    print(f'{nll.item():.2f}, {mse.item():.2f}, {overall_loss.item():.2f}')
    return overall_loss


# In[17]:


def get_model():
    model = ItineraryRNN(input_size=params.input_size,
                         hidden_size=params.hidden_size,
                         output_size=params.output_size,
                         embedding_input=params.embedding_input,
                         embedding_dim=params.embedding_dim,
                         bidirectional=params.bidirectional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr) # torch.optim.Adam(model.parameters(), lr=params.lr)
    criterion = maskedLoss # torch.nn.MSELoss(reduction='mean')
    return model, optimizer, criterion


# In[18]:


model, optimizer, criterion = get_model()


# In[19]:


print(model)
for name, param in model.named_parameters():
    print(f'{name:20} {param.numel()} {list(param.shape)}')
print(f'TOTAL                {sum(p.numel() for p in model.parameters())}')


# In[20]:


#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


# In[21]:


batch_size = params.batch_size
epochs = params.epochs
train_loss = []
valid_loss = []
time_per_epoch = []
print(f'Training cross-validation model for {epochs} epochs')
start = time()

for epoch in range(1, epochs + 1):
    indicator_train = train(model, optimizer, criterion, training_generator)
    train_loss.append(indicator_train)
    print(f'| epoch {epoch:03d} | train loss={indicator_train:.2f}')
    
    indicator_val = validate(model, criterion, valid_generator)
    valid_loss.append(indicator_val)
    print(f'| epoch {epoch:03d} | valid loss={indicator_val:.2f}')
    
    time_per_epoch.append(time() - start)
    print(f'---------------------------------| epoch {epoch:03d} | {time() - start:.2f}s |---------------------------------')
    
    model_name = f'full_network_{epoch}epoch'
    torch.save(model.state_dict(), f'weights/{model_name}.pt')

end = time()
total_time = end - start


# In[22]:


# Logger a TensorBoard
# import matplotlib.pyplot as plt
# plt.plot(training_losses)


# In[23]:


import pickle

measures = {}
measures['training_losses'] = training_losses
measures['train_loss'] = train_loss
measures['valid_loss'] = valid_loss
measures['time_per_epoch'] = time_per_epoch

with open(f'stats/{model_name}_stats.pickle', 'wb') as f:
    pickle.dump(measures, f)


# ***

# ```
# import gc
# for obj in gc.get_objects():
#     try:
#         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#             print(type(obj), obj.size(), a.element_size() * a.num_elements() / 1024**3)
#     except:
#         pass
# ```
