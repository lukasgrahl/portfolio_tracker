import pandas as pd

from pytickersymbols import PyTickerSymbols
from src.pull_data import get_index_yf_tickers

if __name__ == "__main__":
    all_indices = ['EURO STOXX 50', 'FTSE 100', 'OMX Stockholm 30', 'CAC 40', 'DAX', 'MDAX', 'TECDAX', 'IBEX 35',
                   'S&P 500', 'DOW JONES', 'AEX', 'NASDAQ 100']
    index_tickers = ['^STOXX50E', '^FTSE', '^OMX', '^FCHI', '^GDAXI', '^MDAXI', '^TECDAX', '^IBEX', '^GSPC', '^DJI',
                     '^AEX', '^IXIC']
    all_index_dict = dict(zip(all_indices, index_tickers))

    # user input
    sel_ind = 'S&P 500'  # index yfinance ticker
    sel_ind_ticker = [all_index_dict[sel_ind]]

    start = '2015-01-01'
    end = '2022-12-31'
    test_size = .2

    # Load data
    sel_ind_composit_tickers, _, sel_ind_nlargest_tickers, success = get_index_nlargest_composits(sel_ind, n=0)
    print(sel_ind_composit_tickers, _, sel_ind_nlargest_tickers, success)
    df_prices = get_yf_ticker_data([*chain(sel_ind_ticker, sel_ind_nlargest_tickers)], start, end)
    print(df_prices.columns)
    df_prices.reset_index(names='date').to_csv(os.path.join(DATA_DIR, 'prices.csv'))

    # get log return data
    df_rets = np.log(df_prices / df_prices.shift(1)).dropna().copy()

    # get SP500 lead returns for prediction
    df_rets['^GSPC_lead'] = df_rets['^GSPC'].shift(1)
    df_rets.dropna(inplace=True)

    # save returns
    df_rets.reset_index(names='date').to_csv(os.path.join(DATA_DIR, 'returns.csv'))

    # train test split
    train, test = train_test_split(df_rets, test_size)
