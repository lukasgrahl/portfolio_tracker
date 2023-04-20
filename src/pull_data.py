import streamlit as st
import numpy as np
import pandas as pd

import logging

import pandas_datareader as pdread
import yfinance as yf

from itertools import chain

from src.utils import apply_datetime_format
from pytickersymbols import PyTickerSymbols
from src.toml import load_toml
import os
from settings import PROJECT_ROOT

config = load_toml(os.path.join(PROJECT_ROOT, 'config.toml'))
from settings import DATA_DIR


@st.cache_data()
def load_data(sel_ind, sel_ind_ticker, pull_data_start: str, pull_data_end: str,
              n_largest: int = 5, no_internet: bool = False):
    if no_internet:
        df_prices = pd.read_csv(os.path.join(DATA_DIR, 'prices.csv')).rename(columns={'Unnamed: 0': 'index'})
        df_prices['index'] = df_prices['index'].apply(lambda x: apply_datetime_format(x))
        df_prices.set_index('index', inplace=True)

        df_rets = pd.read_csv(os.path.join(DATA_DIR, 'returns.csv')).rename(columns={'Unnamed: 0': 'index'})
        df_rets['index'] = df_rets['index'].apply(lambda x: apply_datetime_format(x))
        df_rets.set_index('index', inplace=True)

        sel_ind_nlargest_tickers = ['MOH.F', 'ASME.F', 'PROSY', 'LOR.F', 'HMI.F']
        lead_name = f"{sel_ind_ticker[0]}_{config['data']['lead_suffix']}"
        return df_prices, df_rets, sel_ind_nlargest_tickers, lead_name

    sel_ind_composit_tickers, _, sel_ind_nlargest_tickers, success = get_index_nlargest_composits(sel_ind, n=n_largest)
    if success <= .8: st.write(f'Market cap was only available for {success * 100: .1f} %  of composits')
    df_prices = get_yf_ticker_data(sel_ind_nlargest_tickers, pull_data_start, pull_data_end)
    df_prices = df_prices.join(get_yf_ticker_data(sel_ind_ticker, pull_data_start, pull_data_end,
                                                  price_kind=['Open', 'High', 'Low', 'Volume', 'Adj Close']))
    df_prices.columns = [item if '_Adj Close' not in item else item[:-10] for item in df_prices.columns]

    # replace zero in volume cols
    x = df_prices[df_prices[f'{sel_ind_ticker[0]}_Volume'] == 0].index
    df_prices.loc[x, f'{sel_ind_ticker[0]}_Volume'] = [1] * len(x)

    # get log returns
    df_rets = np.log(df_prices / df_prices.shift(1)).dropna().copy()

    # get lead variable
    lead_name = f"{sel_ind_ticker[0]}_{config['data']['lead_suffix']}"
    df_rets[lead_name] = df_rets[sel_ind_ticker[0]].shift(-1)
    df_rets = df_rets.dropna()

    return df_prices, df_rets, sel_ind_nlargest_tickers, lead_name


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
            data.append(yf.Ticker(item).info['marketCap'])
            # data.append(pdread.get_quote_yahoo(item)['marketCap'].values[0])
            counter += 1
        except Exception as e:
            logging.info(f'{item} market cap was not found and raised ERROR: {e}')
            data.append(0)
            counter += 0

    df = pd.DataFrame(index=tickers, columns=['market_cap'], data=data)

    market_cap = df.sort_values('market_cap', ascending=False)

    return tickers, market_cap, market_cap.index[:n].values, counter / len(tickers)


def get_index_yf_tickers(index_name: str, data_provider: str = 'yahoo') -> (list, float):
    """
    Get composite yfinance tickers for a given index
    :param data_provider:
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
