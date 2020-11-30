
import pandas as pd
import numpy as np
import pandas_ta as ta

from types import SimpleNamespace

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
    batch_size = 128,
    epochs = 10,
    hidden_size = 512,
    train = True,
    num_layers = 1,
    bidirectional = False,
    lr = 0.001,
    modelname = f'{MODEL_DIR}.pt'
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


#
df = df_test.copy()

df = df.rename_axis('Timestamp').reset_index()

df.isnull().sum()

df.describe()


df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")

df['DLR'] = ta.others.daily_log_return(df.Close)


df_test.ta.log_return()

help(ta.log_return)