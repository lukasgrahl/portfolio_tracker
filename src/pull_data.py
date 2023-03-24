import numpy as np
import pandas as pd
import pandas_datareader as pdread
from yahoo_fin import stock_info as ysi
import yfinance as yf

import os
import time
from itertools import chain

from src.utils import apply_datetime_format
from pytickersymbols import PyTickerSymbols

import streamlit as st


@st.cache_data()
def get_yf_ticker_data(tickers: list, start: str, end: str, price_kind: list = ['Adj Close']) -> pd.DataFrame:
    """
    Pull data from yahoo finance based of yfinance tickers
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
        data = data.drop('Close', axis=1)
        data.columns = list([f'{item}_{x}' for x in data.columns])
        df_prices = df_prices.join(data)


    # get closing price
    cols = [*chain.from_iterable([[item for item in df_prices.columns if price in item] for price in price_kind])]
    df_c = df_prices[cols].copy()
    if price_kind == ['Adj Close']: df_c.columns = [item[:-len(price_kind[0]) - 1] for item in df_c.columns]
    df_c.dropna(inplace=True)

    return df_c


@st.cache_data()
def get_index_nlargest_composits(index_name, n: int = 5) -> (list, pd.DataFrame, list, float):
    """
    Pull list of an index's composits market caps and return n largest composits
    :param n: number of largest composits by market cap
    :return: index tickers, market cap by composit, n largest composits, succes rate on pulling stock tickers
    """

    if n == 0:
        return [], None, [], 1

    tickers, success = get_index_yf_tickers(index_name)
    assert success > .7, f"Less then 70% of {index_name} composits could be retrieved"

    data, counter = [], 0
    for item in tickers:
        # capture error on getting market cap, not recoreded for some stocks
        try:
            data.append(pdread.get_quote_yahoo(item)['marketCap'].values[0])
            counter += 1
        except Exception as e:
            data.append(0)
            counter += 0

    df = pd.DataFrame(index=tickers, columns=['market_cap'], data=data)
    market_cap = df.sort_values('market_cap', ascending=False)

    return tickers, market_cap, market_cap.index[:n].values, counter / len(tickers)


def get_index_yf_tickers(index_name: str, data_provider: str = 'yahoo') -> (list, float):
    """
    Get composite yfinance tickers for a given index
    :param index_name: index name
    :return: index composits tickers, sucess rate on pulling index composite tickers
    """
    tickers = []

    ticker_sym = PyTickerSymbols()
    index_tickers = list(ticker_sym.get_stocks_by_index(index_name))

    counter, index_length = 0, len(index_tickers)
    for item in index_tickers:
        # capture error for getting yahoo ticker symbol, not recorded for some stocks
        try:
            tickers.append(item['symbols'][0][data_provider])
            counter += 1
        except IndexError:
            counter += 0
    return tickers, counter / index_length


@st.cache_data()
def load_csv(file_name: str, path: str, pd_dt_index: str = None, **kwargs) -> pd.DataFrame:
    """
    Loads csv data
    :param file_name:
    :param path:
    :param pd_dt_index: if specified pandas datetime index will be assigned this frequency: "D", "W", "M"
    :param kwargs: pass arguments to pd.read_csv e.g. index_name
    :return: pd.DataFrame
    """
    df = pd.read_csv(os.path.join(path, file_name), **kwargs)
    if pd_dt_index is not None:
        df.index = pd.DatetimeIndex(df.index).to_period(pd_dt_index)
    return df


@st.cache_resource()
def test_res_cache():
    time.sleep(10)
    return True


def train_test_split(df_in: pd.DataFrame, test_size_split: list = [.1]) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits pd.DataFrame alongside axis=0 into train and test sample, assumes most
    recent data to be located on the bottom of the df
    :param df_in:
    :param test_size: list splits: [.1, .1] for a 10%, 80%, 10% split
    :return: test, train
    """
    assert sum(test_size_split) <= 1, "Test size split exceeds 1"
    df = df_in.copy()

    test_size_split.insert(0, 0)
    if sum(test_size_split) < 1: test_size_split.append(1 - sum(test_size_split))

    test_size_split = np.array(test_size_split) * len(df)
    test_size_split = np.array(np.floor(test_size_split), dtype=int)
    test_size_split = np.cumsum(test_size_split)

    dfs_out = []
    for i in range(1, len(test_size_split)):
        dfs_out.append(df.iloc[test_size_split[i-1]: test_size_split[i]])

    return tuple(dfs_out)


## depracted functions
@st.cache_data()
def _get_sp500_n_largest(n: int = 5) -> list:
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
