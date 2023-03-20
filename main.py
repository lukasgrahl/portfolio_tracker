import pandas as pd
import numpy as np
from src.pull_data import get_sp500_n_largest, get_yf_ticker_data
from settings import DATA_DIR

from itertools import chain
import os

if __name__ == '__main__':
    sp500_largest = get_sp500_n_largest()
    np.save(os.path.join(DATA_DIR, 'sp500_largest.npy'), sp500_largest)
    # sp500_largest.to_csv(os.path.join(DATA_DIR, 'sp500_largest.csv'))

    sp500_ticker = ['^GSPC']
    start = '2020-01-01'
    end = '2022-12-31'

    df = get_yf_ticker_data([*chain(sp500_largest, sp500_ticker)], start, end)
    df.reset_index(names='date').to_csv(os.path.join(DATA_DIR, 'prices.csv'))

    # get log returns
    df_rets = np.log(df / df.shift(1)).dropna()
    df_rets.sort_index(inplace=True)

    # get SP500 lead returns for prediction
    df_rets['^GSPC_lead'] = df_rets['^GSPC'].shift(1)
    df_rets.dropna(inplace=True)
    df_rets.index.asfreq = 'D'

    df_rets.reset_index(names='date').to_csv(os.path.join(DATA_DIR, 'returns.csv'))

