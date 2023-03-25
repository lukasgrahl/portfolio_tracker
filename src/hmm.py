import numpy as np

from src.utils import printProgBar
from random import randint
from itertools import chain

from sklearn.mixture import GaussianMixture
import pandas as pd
from src.utils import get_index
import matplotlib.pyplot as plt
import streamlit as st


@st.cache_data()
def get_hmm_features(arr: np.array, ind_ticker: str, cols_list: list, n_largest_stocks):
    x = arr[:, get_index(f'{ind_ticker}_Volume', cols_list)]
    vol_gap = np.diff(x) / x[1:]

    # daily_change
    d_change = ((arr[:, get_index(ind_ticker, cols_list)]
                 - arr[:, get_index(f'{ind_ticker}_Open', cols_list)])
                / arr[:, get_index(f'{ind_ticker}_Open', cols_list)])[1:]

    # fract_high
    frac_high = ((arr[:, get_index(f'{ind_ticker}_High', cols_list)]
                  - arr[:, get_index(f'{ind_ticker}_Open', cols_list)])
                 / arr[:, get_index(f'{ind_ticker}_Open', cols_list)])[1:]

    # fract_low
    frac_low = ((arr[:, get_index(f'{ind_ticker}_Low', cols_list)]
                 - arr[:, get_index(f'{ind_ticker}_Open', cols_list)])
                / arr[:, get_index(f'{ind_ticker}_Open', cols_list)])[1:]

    # n largest stocks
    x = [get_index(item, n_largest_stocks) for item in n_largest_stocks]
    n_largest = [arr[:, x[i]][1:] for i in x]

    # forecast_variable
    forecast = arr[:, get_index(ind_ticker, cols_list)][1:]

    out = [*chain(
        [
            vol_gap,
            d_change,
            frac_high,
            frac_low,
            forecast
        ],
        [item for item in n_largest]
    )]

    cols = [
        *chain(
            [
                'volume_gap',
                'daily_change',
                'fract_high',
                'fract_low',
                'forecast_variable'
            ],
            n_largest_stocks
        )
    ]
    return out, cols


@st.cache_data()
def get_CV_data(data_arr: np.array, cols_list: list, ind_ticker: str, n_largest_stocks: list,
                n_iterations: int, sample_size: tuple = (10, 30)) -> (np.array, list):
    quotes = []

    for i in range(0, n_iterations):
        printProgBar(i, n_iterations - 1)

        # random sample len
        sample_len = randint(sample_size[0], sample_size[1])
        # random sample start
        sample_start = randint(0, len(data_arr) - sample_len)
        subset = data_arr[sample_start: sample_start + sample_len]

        # generate features
        # close_date
        # close_date = np.max(subset[:, get_index('date', cols_list)])

        # volume_gap

        features, cols = get_hmm_features(subset, ind_ticker, cols_list, n_largest_stocks)

        # append
        out_len = len(subset) - 1
        features.insert(0, [i] * out_len)
        out = np.array(features, dtype='object')
        quotes.append(out)

    # user defined, depending on
    ret_cols = [
        *chain(
            [
                'id',
            ],
            cols
        )
    ]
    return quotes, ret_cols


@st.cache_resource()
def get_hmm(X_train, y_train, n_components: int, cov_type: str = 'full', n_int: int = 100, random_state=101):
    mod = GaussianMixture(n_components=n_components, covariance_type=cov_type, n_init=n_int, random_state=random_state)
    mod.fit(X_train, y_train)
    hidden_states = mod.predict(X_train)
    return mod, hidden_states


def get_hidden_states(hidden_states, y_train):
    states = np.zeros((len(y_train), 2))
    states[:, 0] = hidden_states
    states[:, 1] = y_train.reshape(-1)
    states = pd.DataFrame(states, columns=['states', 'rets'])
    statesg = states.groupby('states').agg({'rets': ['mean', 'std', 'count',
                                                     lambda x: np.mean(x) + 1.96 * np.std(x),
                                                     lambda x: np.mean(x) - 1.96 * np.std(x)]}).rename(
        columns={'<lambda_0>': 'conf_lower',
                 '<lambda_1>': 'conf_upper'}
    )
    statesg.columns = statesg.columns.get_level_values(1)
    return states, statesg


def plot_hmm_states(df, y_states, price_col: str, ret_col: str, date_col: str):
    states = set(list(y_states))

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))

    for i in states:
        want = (y_states == i)
        x = df[date_col].iloc[want]
        y = df[price_col].iloc[want]
        ax[0].plot(x, y, '.')
    ax[0].legend(states, fontsize=16)
    ax[0].grid(True)
    ax[0].set_xlabel(date_col, fontsize=16)
    for i in states:
        want = (y_states == i)
        x = df[date_col].iloc[want]
        y = df[ret_col].iloc[want]
        ax[1].plot(x, y, '.')
    ax[1].legend(states, fontsize=16)
    ax[1].grid(True)
    ax[1].set_xlabel("datetime", fontsize=16)
    return fig
