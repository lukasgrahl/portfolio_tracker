import pandas as pd
import numpy as np
from src.pull_data import get_sp500_n_largest, get_yf_ticker_data, load_csv, train_test_split
from settings import DATA_DIR

from itertools import chain
import os

if __name__ == '__main__':
    # user input
    sp500_ticker = ['^GSPC']  # index yfinance ticker
    start = '2020-01-01'
    end = '2022-12-31'
    test_size = .2

    # pull list of sp500 composits
    # sp500_largest = get_sp500_n_largest()
    # np.save(os.path.join(DATA_DIR, 'sp500_largest.npy'), sp500_largest)
    sp500_largest = np.load(os.path.join(DATA_DIR, 'sp500_largest.npy'), allow_pickle=True)

    # get sp500 returns data
    df_prices = get_yf_ticker_data([*chain(sp500_largest, sp500_ticker)], start, end)
    df_prices.reset_index(names='date').to_csv(os.path.join(DATA_DIR, 'prices.csv'))
    df = load_csv(file_name='prices.csv', path=DATA_DIR, index_name='date')

    # get log returns
    # df_rets = get_returns_of_prices(df_prices)
    df_rets = np.log(df_prices / df_prices.shift(1)).dropna()

    # get SP500 lead returns for prediction
    df_rets['^GSPC_lead'] = df_rets['^GSPC'].shift(1)
    df_rets.dropna(inplace=True)

    # save returns
    df_rets.reset_index(names='date').to_csv(os.path.join(DATA_DIR, 'returns.csv'))

    # train test split
    train, test = train_test_split(df_rets, test_size)