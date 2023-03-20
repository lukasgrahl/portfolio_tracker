import pandas as pd
import pandas_datareader as pdread
from yahoo_fin import stock_info as ysi
import yfinance as yf

from src.utils import apply_datetime_format


def get_yf_ticker_data(tickers: list, start: str, end: str, price_kind: str = 'Adj Close'):
    # price_kind: Open, Close, Low, High, Volume
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


def get_sp500_n_largest(n: int = 5):
    # get sp500 composits & market cap
    sp500_tickers = ysi.tickers_sp500()
    df = pd.DataFrame(index=sp500_tickers,
                      columns=['market_cap'],
                      data=[pdread.get_quote_yahoo(item)['marketCap'].values[0] for item in sp500_tickers])
    sp500_largest = df.sort_values('market_cap', ascending=False).index[:n].values
    return sp500_largest
