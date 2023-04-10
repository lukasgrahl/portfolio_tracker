import pandas as pd
import numpy as np
import streamlit as st

from statsmodels.tsa.arima.model import ARIMA
from src.utils import printProgBar

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning


def get_ARMA_test(p, q, train: pd.DataFrame, endog: list, exog: list, sup_warnings: bool = False,
                  vals_only: bool = False):
    if sup_warnings:
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', ValueWarning)
        warnings.simplefilter('ignore', UserWarning)

    if vals_only:
        mod = ARIMA(endog=train[endog].values, exog=train[exog].values, order=(p, 0, q))
    else:
        mod = ARIMA(endog=train[endog], exog=train[exog], order=(p, 0, q))
    res = mod.fit()
    ma_resid = res.resid
    p, q, d = mod.k_ar, mod.k_ma, mod.k_exog
    arima_params = dict(zip(res.param_names, res.params))
    return p, q, d, ma_resid, arima_params, res


@st.cache_resource
def grid_search_arma(p_max: int, q_max: int, data: pd.DataFrame, endog: list, exog: list, **kwargs):
    assert (p_max > 0 and q_max > 0), "AR() and MA() processes both requires at least one lag for the test"
    pq_combinations = [(p, q) for p in range(1, p_max) for q in range(0, q_max)]
    pq_combinations = np.array(pq_combinations).reshape(((p_max - 1) * q_max), 2)
    print(len(pq_combinations))
    out_pqd, out_mod = [], []
    for i in range(0, len(pq_combinations) - 1):
        printProgBar(i, len(pq_combinations)-1, 'ARMA grid search')

        p, q = pq_combinations[i]
        p, q, d, ma_resid, arima_params, mod = get_ARMA_test(p, q, data, endog=endog, exog=exog, **kwargs)
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
