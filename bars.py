import pandas as pd
import numpy as np


def get_timebars(df, time_frame, ffill=True, active=True, show_intervals=False):
    # Get spread
    df['spr'] = df['ask'] - df['bid']
    df['ask_volume'] /= 1000
    df['bid_volume'] /= 1000

    agg = {
        'ask':['max', 'min', 'first', 'last'],
        'bid':['max', 'min', 'first', 'last'],
        'spr':['max', 'min', 'first', 'last'],
        'ask_volume':['sum'],
        'bid_volume':['sum']
    }

    # Resample ticks
    resampled = df.resample(time_frame)
    ndf = resampled.agg(agg)
    ndf[('ticks', 'count')] = resampled['ask'].count()

    # Get trading hours (with more than one tick in half hour).
    assert pd.Timedelta('30 min') % pd.Timedelta(time_frame) == pd.Timedelta(0)
    open = (ndf[('ticks', 'count')]
        .resample('30 min')
        .sum()
        .reindex(ndf.index, method='ffill')) > 0

    if active:
        # keep only open hours
        ndf = ndf[open]
    else:
        ndf['open'] = open

    if show_intervals:
        print("Sessions intervals:")
        print(ndf.index[open != open.shift()])

    if ffill:
        # Forward fill empty rows (no ticks)
        ndf = ndf.ffill()
        # Recompute 'max', 'min', and 'first' of ffill-ed rows (with no ticks)
        row0 = ndf[('ticks', 'count')] == 0
        ndf.loc[row0, [('ask','max'), ('ask','min'), ('ask','first')]] = ndf.loc[row0, ('ask','last')]
        ndf.loc[row0, [('bid','max'), ('bid','min'), ('bid','first')]] = ndf.loc[row0, ('bid','last')]
        ndf.loc[row0, [('ask_volume','sum'), ('bid_volume','sum')]] = 0
        # remove 
    return ndf


def get_tickbars(df, nticks):
    # Keep datetime index as a column
    df = df.reset_index()

    # Get spread
    df['spr'] = df['ask'] - df['bid']
    df['ask_volume'] /= 1000
    df['bid_volume'] /= 1000

    agg = {
        'ask':['max', 'min', 'first', 'last'],
        'bid':['max', 'min', 'first', 'last'],
        'spr':['max', 'min', 'first', 'last'],
        'date':['first', 'last'],
        'ask_volume':['sum'],
        'bid_volume':['sum']
    }

    # Groupby ticks
    resampled = df.groupby(np.arange(len(df.index)) // nticks)
    ndf = resampled.agg(agg)
    ndf[('date', 'delta')] = ndf[('date', 'last')].diff()
    ndf.loc[0, ('date', 'delta')] = ndf.loc[0, ('date', 'last')] - ndf.loc[0, ('date', 'first')]

    return ndf


def add_rolling(df, window, columns, function):

    new_columns = [f'{c}_{window}' for c in columns]
    new_multi_index = pd.MultiIndex.from_product([new_columns, [function]])

    if function == 'first':
        values = df[pd.MultiIndex.from_product([columns, [function]])]
        df[new_multi_index] = values.shift(window) # .fillna(values.iloc[0])
    elif function == 'mean':
        values = df[pd.MultiIndex.from_product([columns, ['last']])]
        df[new_multi_index] = values.rolling(window).mean()
    else:
        values = df[pd.MultiIndex.from_product([columns, [function]])]
        rolling = values.rolling(window)
        method = getattr(rolling, function)
        df[new_multi_index] = method()


def add_rolling_bars(df, tf):
    
    # Basic features
    columns = ['ask', 'bid', 'spr']
    for feature in ['max', 'min', 'first', 'mean']:
        add_rolling(df, tf, columns, feature)

    # Addtional features
    if pd.api.types.is_integer_dtype(df.index):
        add_rolling(df, tf, ['date'], 'first')
        df[(f'date_{tf}', 'delta')] = df[('date', 'last')] - df[('date', 'last')].shift(tf)
        columns = [('ask_volume', 'sum'), ('bid_volume', 'sum')]
        newcols = [(f'ask_volume_{tf}', 'mean'), (f'bid_volume_{tf}', 'mean')]
    else:
        columns = [('ask_volume', 'sum'), ('bid_volume', 'sum'), ('ticks', 'count')]
        newcols = [(f'ask_volume_{tf}', 'mean'), (f'bid_volume_{tf}', 'mean'), (f'ticks_{tf}', 'mean')]

    df[newcols] = df[columns].rolling(tf).mean()

    return df
