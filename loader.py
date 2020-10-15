import os
import pandas as pd


def df_read(path):
    try:
        df = pd.read_hdf(path)
    except FileNotFoundError:
        path = os.path.splitext(path)[0] + '.pkl'
        df = pd.read_pickle(path)
    df.reset_index(drop=True, inplace=True)
    df.set_index('date', inplace=True)
    return df


def df_load(folder, symbol, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if end.day > 1 or end > end.floor('d'):
        range_end = end.floor('d') + pd.offsets.MonthBegin(1)
    else:
        range_end = end
    data_range = pd.date_range(start=start, end=range_end, freq='M')
    filenames = [os.path.join(folder, symbol, f'{symbol}-{d.year}-{d.month:02}.h5') for d in data_range]
    dframes = []
    for f in filenames:
        try:
            df = df_read(f)
            dframes.append(df)
        except FileNotFoundError:
            print('FileNotFound:', f)
    df = pd.concat(dframes)
    df = df.loc[start:end]
    # Pandas df.loc[start_date : end_date] includes both end-points in the result
    if df.index[-1] == end:
        df.drop(df.index[-1], inplace=True)
    return df
