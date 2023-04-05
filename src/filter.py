import numpy as np
import pandas as pd
from itertools import chain

import streamlit
from pypfopt.risk_models import CovarianceShrinkage
from statsmodels.tsa.arima.model import ARIMA
from filterpy.kalman import KalmanFilter
from datetime import datetime, timedelta

from random import randint
import streamlit as st

from src.utils import get_ARlags, get_binary_metric


# @st.cache_data()
def get_ARMA_test(p, q, train: pd.DataFrame, endog: list, exog: list):
    mod = ARIMA(endog=train[endog], exog=train[exog], order=(p, 0, q))
    res = mod.fit()
    ma_resid = res.resid
    p, q, d = mod.k_ar, mod.k_ma, mod.k_exog
    arima_params = dict(zip(res.param_names, res.params))
    return p, q, d, ma_resid, arima_params


def set_up_kalman_filter(p: int, q: int, d: int, xdim: int, zdim: int, data: pd.DataFrame,
                         ma_resid: pd.Series, arima_params: dict, endog: list, exog: list,
                         measurement_noise: float = .01, x0: float = .1, P0: float = .1):
    assert len(endog) == 1, f"The endogenous variable must be unique and cannot be {endog}"

    ma_resid.name = 'ma_resid'
    param_names = list(arima_params.keys())

    # no of parameters
    no_params = p + q + d

    # important ordering or params
    ar_params = [col for col in param_names if 'ar.' in col]  # all AR params
    ma_parmas = [col for col in param_names if 'ma.' in col]  # all MA params
    exo_params = [col for col in param_names if col in exog]  # all exog params
    assert len(ar_params) == p
    assert len(ma_parmas) == q
    assert len(exo_params) == d

    # state variables
    state_vars = [*chain(ar_params, ma_parmas, exo_params)]

    # arima parameters dictionary
    arima_params = {item: arima_params[item] for item in state_vars}

    # get boolean mask for numpy arrays
    ma_bool_mask = [True if "ma.L" in item else False for item in state_vars]
    ar_bool_mask = [True if "ar.L" in item else False for item in state_vars]

    ### Set up Kalman Filter matrices ###

    # T kalman filter transition matrix
    ar_T = np.array([[1 if item == f'ar.L{i}' else 0 for item in state_vars] for i in range(1, p)])
    ma_T = np.array([[1 if item == f'ma.L{i}' else 0 for item in state_vars] for i in range(1, q + 1)])
    exo_T = np.array([[1 if exo == name else 0 for name in state_vars] for exo in exo_params])

    T = np.array([list(arima_params.values())])
    if p > 1: T = np.append([T], [ar_T], axis=1)[0]  # .reshape(no_params, 2)
    if q > 0: T = np.append([T], [ma_T], axis=1)[0]
    if d > 0: T = np.append([T], [exo_T], axis=1)[0]

    # H measurement noise
    H = np.diag([measurement_noise] * zdim)

    # Z measurement function: measurement -> state space
    Z = np.diag([1] * zdim)

    # zs observation matrix
    ar_df = get_ARlags(data[endog[0]], p, ret_org_ser=False)
    ma_df = get_ARlags(ma_resid, q, ret_org_ser=False)
    exo_df = data[exog].copy()

    df = ar_df.join(ma_df)
    df = df.join(exo_df)
    df = df.dropna()

    zs = df.values.reshape((len(df), no_params, 1))
    zs[:, ma_bool_mask] = np.zeros((len(df), q, 1))

    # Q process noise matrix
    Q = CovarianceShrinkage(df, returns_data=True, log_returns=True).ledoit_wolf().values

    # x, P set initial state and covariance values
    x0 = np.zeros([xdim]) + x0
    x0 = x0.reshape(xdim, 1)
    P0 = np.diag([P0] * xdim)

    return T, Q, Z, H, x0, P0, zs, state_vars, df.index


def kalman_filter(xdim, zdim, p, q, d, x0, P0, zs, T, Q, Z, H, state_vars):
    no_params = p + q + d

    # initialise kalaman filter object
    kfilter = KalmanFilter(xdim, zdim)
    kfilter.F = T
    kfilter.Q = Q
    kfilter.H = Z
    kfilter.R = H

    # set starting values
    kfilter.x = x0
    kfilter.P = P0

    # set ouput lists
    X_out, X_pred, P_out, P_pred, LL_out = [], [], [], [], []

    # z starting value
    z = zs[0]
    zs = zs[1:]

    for i in range(0, len(zs) + 1):

        # kalman predict step
        kfilter.predict()
        X_pred.append(kfilter.x)
        P_pred.append(kfilter.P)

        # kalman update step
        kfilter.update(z)

        # get mask for updating the MA component
        ma_bool_mask = [True if "ma.L" in item else False for item in state_vars]
        ma_partial_mask = [[True if item == f'ma.L{ix}' else False for item in state_vars] for ix in range(1, q + 1)]

        # update MA components with lagged prediction error ahead
        if q > 0:
            if i + q - 1 + 1 < len(zs):
                residual = zs[i + 1, ma_bool_mask][0] - kfilter.x[0]
                for iz in range(0, q):
                    zs[i + iz + 1, ma_partial_mask[iz]] = residual

        # set nex iterations z
        if i + 1 < len(zs): z = zs[i + 1]

        X_out.append(kfilter.x)
        P_out.append(kfilter.P)
        LL_out.append(kfilter.log_likelihood)

    # one time period ahead prediction
    kfilter.predict()
    X_pred.append(kfilter.x)
    P_pred.append(kfilter.P)

    X_out = np.array(X_out)
    P_out = np.array(P_out)
    X_pred = np.array(X_pred)
    P_pred = np.array(P_pred)
    LL_out = np.array(LL_out)

    return X_out, P_out, X_pred, P_pred, LL_out


def run_kalman_filter(endog: list, exog: list, df_rets: pd.DataFrame, measurement_noise: float):
    # this should later be replaced by an automatic ARMA pq definition
    p, q = 2, 1
    # get arima output
    p, q, d, ma_resid, arima_params = get_ARMA_test(p, q, df_rets, endog, exog)

    ####### Kalman Filter #####
    xdim = p + d + q
    zdim = xdim
    # set up filter
    T, Q, Z, H, x0, P0, zs, state_vars, zs_index = set_up_kalman_filter(p, q, d, xdim, zdim, df_rets, ma_resid,
                                                                        arima_params, endog, exog, measurement_noise)
    # run filter
    X_out, P_out, X_pred, P_pred, LL_out = kalman_filter(xdim, zdim, p, q, d, x0, P0, zs, T, Q, Z, H, state_vars)

    # get output as pd.DataFrame
    df_xtrue = df_rets[endog].loc[zs_index].copy()
    ind = pd.DatetimeIndex([str(item) for item in zs_index])
    df_xtrue = pd.DataFrame(df_xtrue.values, index=ind, columns=endog)
    df_xfilt = pd.DataFrame(X_out[:, 0], index=ind, columns=[f'{endog[0]}_filter'])

    ind = pd.DatetimeIndex([*chain(
        [str(item) for item in zs_index],
        [str(datetime.now().date() + timedelta(days=1))]
    )])
    df_xpred = pd.DataFrame(X_pred[:, 0], index=ind, columns=[f'{endog[0]}_pred'])

    return df_xtrue, df_xpred, df_xfilt


@st.cache_resource
def get_kalman_cv(df_rets: pd.DataFrame, endog: list, exog: list, measurement_noise: float,
                  cv_index_len: int, sample_len_weeks: int, no_samples: int = 20):
    cv_output = []
    for i in range(0, no_samples):
        cv_start = randint(0, (cv_index_len - sample_len_weeks * 5))  # take
        cv_end = cv_start + (sample_len_weeks * 5)  #

        df_rets_sel = df_rets.iloc[cv_start: cv_end].copy()

        df_xtrue, df_xpred, df_xfilt = run_kalman_filter(endog, exog, df_rets_sel, measurement_noise)
        df_xtrue = df_xtrue.iloc[5:]
        df_xpred = df_xpred.iloc[5:]
        cv_output.append([df_xtrue.values.reshape(-1), df_xpred.iloc[:-1].values.reshape(-1)])

    cv_output = np.array(cv_output)
    conf_mat, roc_score = get_binary_metric(np.concatenate(cv_output[:, 0, :]), np.concatenate(cv_output[:, 1, :]),
                                            cut_off=0)
    return conf_mat, roc_score
