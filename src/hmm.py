import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from random import randint
from itertools import chain
from copy import deepcopy

from sklearn.mixture import GaussianMixture
from src.utils import get_index, printProgBar, train_test_split
from sklearn.preprocessing import scale


@st.cache_resource()
def run_hmm(data: pd.DataFrame, sel_ind_ticker: list, lead_name: str, sel_ind_nlargest_ticker: list,
            hmm_states: int, hmm_init: int, cv_samples: int, train_test_size: list = [.9]):
    train, test = train_test_split(data, test_size_split=train_test_size)

    # get test data
    arr_test, test_cols = get_hmm_features(arr=test.values, ind_ticker=sel_ind_ticker[0], lead_var=lead_name,
                                           cols_list=list(test.columns),
                                           n_largest_stocks=list(sel_ind_nlargest_ticker))
    arr_test = np.array(arr_test, dtype=float)
    arr_test = arr_test.transpose()

    # get train data
    arr_train, train_cols = get_hmm_features(arr=train.values, ind_ticker=sel_ind_ticker[0], lead_var=lead_name,
                                             cols_list=list(train.columns),
                                             n_largest_stocks=list(sel_ind_nlargest_ticker))
    arr_train = np.array(arr_train, dtype=float)
    arr_train = arr_train.transpose()

    # get cross validation train data
    arr_train_cv, train_cols_cv = get_CV_data(data_arr=train.values, cols_list=list(train.columns),
                                              ind_ticker=sel_ind_ticker[0], lead_var=lead_name,
                                              n_largest_stocks=list(sel_ind_nlargest_ticker),
                                              n_iterations=cv_samples)
    arr_train_cv = np.concatenate(arr_train_cv, axis=1)
    arr_train_cv = arr_train_cv.transpose()

    # get scaled data
    arr_train_cv_s = np.column_stack([scale(arr_train_cv[:, i]) for i in range(arr_train_cv.shape[1])])
    arr_train_s = np.column_stack([scale(arr_train[:, i]) for i in range(arr_train.shape[1])])
    arr_test_s = np.column_stack([scale(arr_test[:, i]) for i in range(arr_test.shape[1])])

    # get train and test sets
    X_test = arr_test_s[:, ~get_index('forecast_variable', test_cols, True)].copy()
    y_test = arr_test_s[:, get_index('forecast_variable', test_cols, True)].copy()

    X_train = arr_train_s[:, ~get_index('forecast_variable', train_cols, True)].copy()
    y_train = arr_train_s[:, get_index('forecast_variable', train_cols, True)].copy()

    X_train_cv = arr_train_cv_s[:, ~get_index('forecast_variable', train_cols_cv, True)].copy()
    y_train_cv = arr_train_cv_s[:, get_index('forecast_variable', train_cols_cv, True)].copy()

    # train model
    mod, train_cv_states = get_hmm(X_train_cv, y_train_cv, n_components=hmm_states, n_int=hmm_init)

    # get cv_states
    cv_states, cv_statesg = get_hidden_states(train_cv_states, arr_train_cv[:, get_index('forecast_variable',
                                                                                         train_cols_cv, True)])

    # get test df
    y_test_df = mod.predict(X_test)
    X_test_df = pd.DataFrame(X_test, columns=train_cols_cv[:-1])
    X_test_df[f'{sel_ind_ticker[0]}_price'] = test[f'{sel_ind_ticker[0]}_price'].iloc[1:].values
    X_test_df[f'{sel_ind_ticker[0]}'] = test[f'{sel_ind_ticker[0]}'].iloc[1:].values
    X_test_df['date'] = list(test.index)[1:]

    # get train df
    y_train_df = mod.predict(X_train)
    X_train_df = pd.DataFrame(X_train, columns=train_cols[:-1])
    X_train_df[f'{sel_ind_ticker[0]}_price'] = train[f'{sel_ind_ticker[0]}_price'].iloc[1:].values
    X_train_df[f'{sel_ind_ticker[0]}'] = train[f'{sel_ind_ticker[0]}'].iloc[1:].values
    X_train_df['date'] = list(train.index)[1:]

    return mod, train_cv_states, cv_states, cv_statesg, (X_test, y_test, X_test_df, y_test_df), \
        (X_train, y_train, X_train_df, y_train_df), (X_train_cv, y_train_cv)


@st.cache_data()
def get_hmm_features(arr: np.array, ind_ticker: str, lead_var: str, cols_list: list, n_largest_stocks) -> (list, list):
    arr = deepcopy(arr)
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
    forecast = arr[:, get_index(lead_var, cols_list)][1:]

    ret = [*chain(
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
    return ret, cols


@st.cache_data()
def get_CV_data(data_arr: np.array, cols_list: list, ind_ticker: str, lead_var: str, n_largest_stocks: list,
                n_iterations: int, sample_size: tuple = (10, 30)) -> (list, list):
    quotes = []

    for i in range(0, n_iterations):
        printProgBar(i, n_iterations - 1)

        # random sample len
        sample_len = randint(sample_size[0], sample_size[1])
        sample_start = randint(0, len(data_arr) - sample_len)
        subset = deepcopy(data_arr[sample_start: sample_start + sample_len])

        # generate features
        features, cols = get_hmm_features(subset, ind_ticker, lead_var, cols_list, n_largest_stocks)
        assert cols == cols, 'get_hmm_feature cols have changed'
        out = np.array(features, dtype='object')
        quotes.append(out)

    # user defined, depending on
    ret_cols = deepcopy(cols)

    return quotes, ret_cols


@st.cache_resource()
def get_hmm(X_train, y_train, n_components: int, cov_type: str = 'full',
            n_int: int = 100, random_state=101) -> (GaussianMixture, np.array):
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


def plot_hmm_states(df, y_states, price_col: str, ret_col: str, date_col: str, test_ind: int, font_size: int = 10):
    states = set(list(y_states))
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))

    for i in states:
        want = y_states == i
        x = df[date_col].iloc[want]
        y = df[price_col].iloc[want]
        ax[0].plot(x, y, '.', label=f"state {i}")

    ax[0].vlines(x=df.iloc[test_ind][date_col], ymin=df[price_col].min() * .95,
                 ymax=df[price_col].max() * 1.05, label='test_set', linewidth=.5, color='black', linestyle='-')
    ax[0].legend(fontsize=font_size)
    ax[0].grid(True)
    ax[0].set_xlabel(date_col, fontsize=font_size)

    for i in states:
        want = y_states == i
        x = df[date_col].iloc[want]
        y = df[ret_col].iloc[want]
        ax[1].plot(x, y, '.', label=f"state {i}")

    ax[1].vlines(x=df.iloc[test_ind][date_col], ymin=df[ret_col].min() * .95,
                 ymax=df[ret_col].max() * 1.05, label='test_set', linewidth=.5, color='black', linestyle='-')
    ax[1].legend(fontsize=font_size)
    ax[1].grid(True)
    ax[1].set_xlabel("datetime", fontsize=font_size)

    plt.tight_layout()
    return fig
