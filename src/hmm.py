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


def run_hmm(data: pd.DataFrame, sel_ind_ticker: list, lead_name: str, sel_ind_nlargest_ticker: list,
            hmm_states: int, hmm_init: int, cv_samples: int, train_test_size: float = .9, **kwargs):
    """
    Function that draws from the below to train and test an entire hidden markov model end to end

    :param data: data frame containing all relevant columns
    :param sel_ind_ticker: ticker of the index chosen for forecasting
    :param lead_name: name of the sel_ind_ticker as lead (t+1)
    :param sel_ind_nlargest_ticker: list of tickers of the n largest composits of that ticker
    :param hmm_states: number of hidden markov states to be identified
    :param hmm_init: init argument for the GaussianMixture
    :param cv_samples: number of cross validation samples
    :param train_test_size: train data set size
    :param kwargs: cv_sample_sizes = tuple(10, 30) - set tuple of intervals from which CV samplesize is randomly drawn
    :return:
    """
    train, test = train_test_split(data, test_size_split=[train_test_size])

    # get test data
    arr_test, test_cols = get_hmm_features(arr=test.values, target_var_t=sel_ind_ticker[0], target_var_lead=lead_name,
                                           cols_list=list(test.columns),
                                           exogenous_vars=list(sel_ind_nlargest_ticker))
    arr_test = np.array(arr_test, dtype=float)
    arr_test = arr_test.transpose()

    # get train data
    arr_train, train_cols = get_hmm_features(arr=train.values, target_var_t=sel_ind_ticker[0],
                                             target_var_lead=lead_name,
                                             cols_list=list(train.columns),
                                             exogenous_vars=list(sel_ind_nlargest_ticker))
    arr_train = np.array(arr_train, dtype=float)
    arr_train = arr_train.transpose()

    # get cross validation train data
    arr_train_cv, train_cols_cv = get_CV_data(data_arr=train.values, cols_list=list(train.columns),
                                              target_var_t=sel_ind_ticker[0], target_var_lead=lead_name,
                                              exogenous_vars=list(sel_ind_nlargest_ticker),
                                              no_samples=cv_samples, **kwargs)
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


def get_hmm_features(arr: np.array, target_var_t: str, target_var_lead: str, cols_list: list, exogenous_vars) \
        -> (list, list):
    """
    Function that computes model features of exogenous variables for the get_CV_data, written in numpy for speed
    This function is highly specific to the underlying data, whereas all other functions are as generic as possible.
    If other features need to be calculated these should be defined in here, therefore this function also assigns
    variable names.

    :param arr: array of data
    :param target_var_t: target variable at time t
    :param target_var_lead: target variable with lead at time t
    :param cols_list: list of columns as they appear in arr
    :param exogenous_vars: list of exogenous variables not needing any further transformation
    :return: arr plus newly computed features, list of columns as they appear in arr
    """

    new_cols = ['volume_gap', 'daily_change', 'fract_high', 'fract_low', 'forecast_variable']

    arr = deepcopy(arr)
    x = arr[:, get_index(f'{target_var_t}_Volume', cols_list)]
    vol_gap = np.diff(x) / x[1:]

    ## the following can be extended in accordance with new_cols and column suffix names ##
    # daily_change
    d_change = ((arr[:, get_index(target_var_t, cols_list)]
                 - arr[:, get_index(f'{target_var_t}_Open', cols_list)])
                / arr[:, get_index(f'{target_var_t}_Open', cols_list)])[1:]

    # fract_high
    frac_high = ((arr[:, get_index(f'{target_var_t}_High', cols_list)]
                  - arr[:, get_index(f'{target_var_t}_Open', cols_list)])
                 / arr[:, get_index(f'{target_var_t}_Open', cols_list)])[1:]

    # fract_low
    frac_low = ((arr[:, get_index(f'{target_var_t}_Low', cols_list)]
                 - arr[:, get_index(f'{target_var_t}_Open', cols_list)])
                / arr[:, get_index(f'{target_var_t}_Open', cols_list)])[1:]

    # n largest stocks
    x = [get_index(item, exogenous_vars) for item in exogenous_vars]
    n_largest = [arr[:, x[i]][1:] for i in x]

    # forecast_variable
    forecast = arr[:, get_index(target_var_lead, cols_list)][1:]

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

    cols = [*chain(new_cols, exogenous_vars)]
    return ret, cols


@st.cache_data()
def get_CV_data(data_arr: np.array, cols_list: list, target_var_t: str, target_var_lead: str, exogenous_vars: list,
                no_samples: int, cv_sample_sizes: tuple = (10, 30)) -> (list, list):
    """
    Function to obtain cross validated training data for a hidden markov model, written exclusively in numpy for speed
    of computation
    :param data_arr: np.array of data, can be multidimensional
    :param cols_list: list of columns as contained in the numpy array
    :param target_var_t: str name of the target variable at time t, must be contained in cols_list
    :param target_var_lead: target variable with lead of t+1
    :param exogenous_vars: list of exogenous variables that need no further transformation
    :param no_samples: number of samples, cross validation folds
    :param cv_sample_sizes: interval in between a random sample size will be assigned
    :return: list containing individuals samples as np.array, list of columns of the np.array
    """
    assert len(cols_list) == data_arr.shape[1], "Columns list does not correspond to number of columns in data"
    assert target_var_t in cols_list, f"{target_var_t} is not in cols_list"
    assert target_var_lead in cols_list, f"{target_var_lead} is not in cols_list"

    cv_out = []

    for i in range(0, no_samples):
        printProgBar(i, no_samples - 1, 'HMM cross validation')

        # random sample len
        sample_len = randint(cv_sample_sizes[0], cv_sample_sizes[1])
        sample_start = randint(0, len(data_arr) - sample_len)
        subset = deepcopy(data_arr[sample_start: sample_start + sample_len])

        # generate features
        features, cols = get_hmm_features(subset, target_var_t, target_var_lead, cols_list, exogenous_vars)
        assert cols == cols, 'get_hmm_feature cols have changed'
        out = np.array(features, dtype='object')
        cv_out.append(out)

    return cv_out, cols


@st.cache_resource()
def get_hmm(X_train, y_train, n_components: int, cov_type: str = 'full',
            n_int: int = 100, random_state=101) -> (GaussianMixture, np.array):
    """
    Function training a hidden markov model
    :param X_train: exogenous variables
    :param y_train: endogenous variable
    :param n_components: number of hidden states
    :param cov_type: sklearn GaussianMixture cov_type
    :param n_int: initialisation parameter for GaussianMixture
    :param random_state:
    :return: GaussianMixture model, hidden states
    """
    mod = GaussianMixture(n_components=n_components, covariance_type=cov_type, n_init=n_int, random_state=random_state)
    mod.fit(X_train, y_train)
    hidden_states = mod.predict(X_train)
    return mod, hidden_states


def get_hidden_states(hidden_states: np.array, returns: np.array) -> (pd.DataFrame, pd.DataFrame):
    """
    Creates summary statistics for hidden states
    :param hidden_states: hidden-states prediction
    :param returns: returns data
    :return: df containing states and returns in two columns, descriptive statistics by state
    """
    states = np.zeros((len(returns), 2))
    states[:, 0] = hidden_states
    states[:, 1] = returns.reshape(-1)
    states = pd.DataFrame(states, columns=['states', 'rets'])
    statesg = states.groupby('states').agg({'rets': ['mean', 'std', 'count',
                                                     lambda x: np.mean(x) + 1.96 * np.std(x),
                                                     lambda x: np.mean(x) - 1.96 * np.std(x)]}).rename(
        columns={'<lambda_0>': 'conf_lower',
                 '<lambda_1>': 'conf_upper'}
    )
    statesg.columns = statesg.columns.get_level_values(1)
    return states, statesg


def plot_hmm_states(df, y_states: np.array, price_col: str, ret_col: str, date_col: str,
                    test_ind: int, font_size: int = 10) -> plt.figure:
    """
    Function that plots returns and prices coloured according to their predicted markov state
    :param df: dataframe of returns and prices
    :param y_states: array of states with same axis=0 dimension as df - used for filtering
    :param price_col: column containing prices
    :param ret_col: column containing returns
    :param date_col: column containing date and time
    :param test_ind: start index of test data
    :param font_size: plot font size
    :return: plot
    """

    assert len(df) == len(y_states), "y_states does not correspond to df dimension"
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
