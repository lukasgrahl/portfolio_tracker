import numpy as np
import pandas as pd
import pandas_datareader as pdread
from yahoo_fin import stock_info as ysi
import yfinance as yf

import os
import time

from src.utils import apply_datetime_format

import streamlit as st


@st.cache_data()
def get_yf_ticker_data(tickers: list, start: str, end: str, price_kind: str = 'Adj Close') -> pd.DataFrame:
    """
    Pull data from yahoo finance from tickers
    :param tickers: list of yfinance ticker
    :param start: start date in '%Y-%m-%d' format
    :param end: end date in '%Y-%m-%d' format
    :param price_kind: choose between Open, Close, Low, High, Volume
    :return: df of tickers and price_kind
    """
    df_prices = pd.DataFrame()
    df_prices.index = pd.date_range(start, periods=(
            apply_datetime_format(end, '%Y-%m-%d') - apply_datetime_format(start, '%Y-%m-%d')).days)

    for item in tickers:
        data = yf.download(item, start, end)
        data.columns = list([f'{item}_{x}' for x in data.columns])
        df_prices = df_prices.join(data)

    # get closing price
    df_c = df_prices[[item for item in df_prices.columns if price_kind in item]].copy()
    df_c.columns = [item[:-len(price_kind) - 1] for item in df_c.columns]
    df_c.dropna(inplace=True)

    return df_c


@st.cache_data()
def get_sp500_n_largest(n: int = 5) -> list:
    """
    Pull list of SP 500 composits and their market cap to obtian n largest
    :param n: number of largest composits by market cap
    :return: list n yfinance tickers
    """
    # get sp500 composits & market cap
    sp500_tickers = ysi.tickers_sp500()
    df = pd.DataFrame(index=sp500_tickers,
                      columns=['market_cap'],
                      data=[pdread.get_quote_yahoo(item)['marketCap'].values[0] for item in sp500_tickers])
    sp500_largest = df.sort_values('market_cap', ascending=False).index[:n].values
    return sp500_largest


# @st.cache_data()
def load_csv(file_name: str, path: str, time_period: str = None, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path, file_name), **kwargs)
    if time_period is not None:
        df.index = pd.DatetimeIndex(df.index).to_period(time_period)
    return df


@st.cache_resource()
def test_res_cache():
    time.sleep(10)
    return True


# def get_returns_of_prices(prices: pd.DatFrame):
#     return np.log(prices / prices.shift(1)).dropna().sort_index()

def train_test_split(df_in: pd.DataFrame, test_size) -> (pd.DataFrame, pd.DataFrame):
    df = df_in.copy()
    test_ind = int(len(df) * test_size)
    # df['test_set'] = [*chain([False] * (len(df) - test_ind), [True] * test_ind)]
    return df.iloc[:test_ind], df[test_ind:]
