import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import matplotlib.pyplot as plt

import os

from settings import DATA_DIR

from src.pull_data import get_yf_ticker_data, get_sp500_n_largest, load_csv, test_res_cache
from src.filter import get_ARMA_test, set_up_kalman_filter, kalman_filter

if __name__ == '__main__':
    sp500_ticker = ['^GSPC']
    start = '2020-01-01'
    end = '2022-12-31'

    # streamlit setup
    st.set_page_config(page_title='A binary guide to the S&P 500', layout='wide')

    # Part 1
    st.header('SP 500 price overview')
    st.write(
       f"""
        The past daily prices from {start} to {end}
        """
    )
    a = st.columns(1)

    clear_cache = st.button('Clear cache')

    # Load data
    sp500_largest = get_sp500_n_largest()
    df_prices = load_csv('prices.csv', DATA_DIR, time_period='D', index_col='date')
    df_rets = np.log(df_prices / df_prices.shift(1)).dropna().copy()

    # ARIMA
    endog = ['^GSPC']
    exog = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN']
    p, q = 2, 1

    # get SP500 lead returns for prediction
    df_rets[f'{endog[0]}_lead'] = df_rets[endog[0]].shift(1)
    df_rets.dropna(inplace=True)

    # get arima output
    p, q, d, ma_resid, arima_params = get_ARMA_test(p, q, df_rets, endog, exog)

    # set up filter
    xdim = p + d + q
    zdim = xdim

    T, Q, Z, H, x0, P0, zs, state_vars = set_up_kalman_filter(p, q, d, xdim, zdim,
                                                              df_rets, ma_resid, arima_params, endog, exog)
    X_out, P_out, X_pred, P_pred, LL_out = kalman_filter(xdim, zdim, p, q, d, x0, P0, zs, T, Q, Z, H, state_vars)

    # get kalman prediction into df
    df_xpred = pd.DataFrame(data=X_pred[:, 0], index=df_rets.index[:len(X_pred)], columns=['predicted'])
    df_xpred = df_xpred.join(df_rets[endog])
    # test cache
    # _ = test_res_cache()

    # plots
    fig1 = px.line(df_prices[sp500_ticker].values)
    fig2 = px.line(X_pred[:, 0])
    # fig2 = px.line(x=df_xpred.index, y=df_xpred['predicted'])
    # fig2.add_scatter(x=df_rets.index, y=df_rets[endog[0]])
    fig3 = px.line(df_rets[endog[0]].values)

    # fill streamlit page
    st.plotly_chart(fig1, theme='streamlit', use_container_width=True)

    st.header('Forecasting output')
    tab1, tab2 = st.tabs(["Kalman Filter", "ARIMA"])
    with tab1:
        st.line_chart(df_xpred)
    with tab2:
        st.plotly_chart(fig3, theme='streamlit', use_container_width=True)

    if clear_cache:
        st.cache_data.clear()