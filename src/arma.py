import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning


def get_ARMA(p, q, data: pd.DataFrame, endog: list, exog: list, sup_warnings: bool = False,
             vals_only: bool = False) -> (int, int, int, np.array, dict, np.array):
    """
    runs statsmodels.tsa.model.ARIMA and returns output
    :param p: AR lags minimum 1
    :param q: MA lags
    :param data: dataframe containing time series and potential exogenous variables
    :param endog: endogenous variable "the time series" names
    :param exog: exogenenous variable names
    :param sup_warnings: should ARIMA warnings be supressed, this is usefull for the grid search where some iterations
    necessarily will fail to converge in probability
    :param vals_only: runs the function on np.arrays only (speeds up the process)
    :return: AR lags, MA lags, no of exogenous variables, moving average residuals, dictionary of coefficients,
    model results (model non-forecasted time series value estimate)
    """
    if sup_warnings:
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', ValueWarning)
        warnings.simplefilter('ignore', UserWarning)

    if vals_only:
        mod = ARIMA(endog=data[endog].values, exog=data[exog].values, order=(p, 0, q))
    else:
        mod = ARIMA(endog=data[endog], exog=data[exog], order=(p, 0, q))
    res = mod.fit()
    ma_resid = res.resid
    p, q, d = mod.k_ar, mod.k_ma, mod.k_exog
    arima_params = dict(zip(res.param_names, res.params))
    return p, q, d, ma_resid, arima_params, res


@st.cache_resource
def grid_search_arma(p_max: int, q_max: int, data: pd.DataFrame, endog: list,
                     exog: list, **kwargs) -> (int, int, int, np.array, dict, ARIMA):
    """
    Runs a grid search across a defined AR(p) and MA(q) space to evaluate wich specification is within 10% range of the
    lowes BIC. Out of this 10% range the simples model specification will be chosen.
    :param p_max: range of AR lags
    :param q_max: range of MA lags
    :param data: data to run the ARMA model on
    :param endog: list of endogenous variables
    :param exog: list of exogenous variables
    :param kwargs: sup_warning, to surpress ARMA warnings for non-convergence (this will naturally occur)
    :return: optiomal p, optimal q, no of exogenous variables, moving average residuals, arma coefficients, arma model
    """
    assert (p_max > 0 and q_max > 0), "AR() and MA() processes both requires at least one lag for the test"
    pq_combinations = [(p, q) for p in range(1, p_max) for q in range(0, q_max)]
    pq_combinations = np.array(pq_combinations).reshape(((p_max - 1) * q_max), 2)
    print(len(pq_combinations))
    out_pqd, out_mod = [], []
    for i in range(0, len(pq_combinations) - 1):
        p, q = pq_combinations[i]
        p, q, d, ma_resid, arima_params, mod = get_ARMA(p, q, data, endog=endog, exog=exog, **kwargs)
        out_pqd.append([p, q, d, mod.bic])
        out_mod.append([p, q, d, ma_resid, arima_params, mod])

    out_pqd = np.array(out_pqd)
    df_out = pd.DataFrame(np.array(out_pqd)[:, [True, True, False, True]], columns=['p', 'q', 'bic'])
    df_out = pd.pivot_table(df_out, values='bic', index='p', columns='q')
    bic_min = df_out.min().min()
    df_out = df_out / bic_min - 1
    df_out = df_out.stack().reset_index()

    df_sel = df_out[df_out[0] > -.01]
    p_best, q_best = df_out.loc[(df_sel.p + df_sel.q).idxmin()].iloc[:2]
    p_best, q_best = int(p_best), int(q_best)

    ind = [out_mod.index(item) for item in out_mod if (item[0] == p_best and item[1] == q_best)][0]
    d, ma_resid, arma_params, arma_mod = out_mod[ind][2], out_mod[ind][3], out_mod[ind][4], out_mod[ind][5]

    return p_best, q_best, d, ma_resid, arma_params, arma_mod
