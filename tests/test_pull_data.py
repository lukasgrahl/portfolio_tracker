import pandas as pd

from pytickersymbols import PyTickerSymbols
from src.pull_data import get_index_yf_tickers

if __name__ == "__main__":
    ticker_sym = PyTickerSymbols()

    all_tickers = ticker_sym.get_all_indices()
    all_indices = ['EURO STOXX 50',  'FTSE 100', 'OMX Stockholm 30', 'CAC 40', 'DAX', 'MDAX', 'TECDAX', 'IBEX 35',
                   'S&P 500', 'DOW JONES', 'Switzerland 20', 'AEX', 'NASDAQ 100']
    index_tickers = ['^STOXX50E', '^FTSE', '^OMX', '^FCHI', '^GDAX', '^MDAXI', '^TECDAX', '^IBEX', '^GSPC', '^DJI', '^SPI20.SW', '^AEX', 'IXIC']


    # for i in range(0, len(all_indices)):
    #     t, _ = get_index_yf_tickers(all_indices[i])
    #     print(all_indices[i], '   ', _, '\n')

    from datetime import datetime, timedelta
    analysis_time = 100

    end = datetime.now().date()
    start = end - analysis_time

    print('x')
