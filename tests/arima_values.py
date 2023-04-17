p = 1
d = 0
q = 1

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from src.pull_data import load_data
from src.toml import load_toml
from settings import PROJECT_ROOT
from src.arma import grid_search_arma
from src.utils import printProgBar
from datetime import datetime

if __name__ == '__main__':
    config = load_toml(os.path.join(PROJECT_ROOT, 'config.toml'))
    endog = '^STOXX50E'
    DF_PRICES, DF_RETS, SEL_IND_NLARGEST_TICKERS, LEAD_NAME = load_data('CAC 40', [endog], '2022-01-01', '2023-01-01',
                                                                        n_largest=config['data']['n_largest_composits'],
                                                                        no_internet=True)

    p, q, d, residuals, params, mod = grid_search_arma(5, 1, DF_RETS.iloc[-50:], endog=[endog],
                                                  exog=SEL_IND_NLARGEST_TICKERS, sup_warnings=True)

    print(p, q)
