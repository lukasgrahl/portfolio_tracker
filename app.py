import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score

from itertools import chain
from datetime import datetime, timedelta

from src.pull_data import get_yf_ticker_data, get_index_nlargest_composits, load_csv, test_res_cache
from src.filter import get_ARMA_test, set_up_kalman_filter, kalman_filter

if __name__ == '__main__':
    # streamlit setup
    st.set_page_config(page_title='A binary guide to the S&P 500', layout='wide')

    # currently supported indices
    all_indices = ['EURO STOXX 50', 'FTSE 100', 'OMX Stockholm 30', 'CAC 40', 'DAX', 'MDAX', 'TECDAX', 'IBEX 35',
                   'S&P 500', 'DOW JONES', 'AEX', 'NASDAQ 100']
    index_tickers = ['^STOXX50E', '^FTSE', '^OMX', '^FCHI', '^GDAXI', '^MDAXI', '^TECDAX', '^IBEX', '^GSPC', '^DJI', '^AEX', '^IXIC']
    all_index_dict = dict(zip(all_indices, index_tickers))

    # streamlit side bar
    with st.sidebar:
        # select index
        sel_ind = st.selectbox('What index to analyse?', tuple(all_indices))
        sel_ind_ticker = [all_index_dict[sel_ind]]

        # select time window
        analysis_time = st.select_slider(
            'How many weeks would you like the analysis to run on?',
            options=list(range(10, 120, 10)))
        dt_end = datetime.now().date()
        dt_start = dt_end - timedelta(weeks=analysis_time)
        start = str(dt_start)
        end = str(dt_end)

        st.write(f'The analysis will run for {analysis_time} weeks from {start} to {end}')

        clear_cache = st.button('Clear cache')


    # first part
    st.header(f'{sel_ind} price overview')
    st.write(
        f"""
            The past daily prices from {start} to {end}
            """
    )

    # Load data
    sel_ind_composit_tickers, _, sel_ind_nlargest_tickers, success = get_index_nlargest_composits(sel_ind)
    st.write(f'{success} success on pulling market cap')
    df_prices = get_yf_ticker_data([*chain(sel_ind_ticker, sel_ind_nlargest_tickers)], start, end)

    # get log return data
    df_rets = np.log(df_prices / df_prices.shift(1)).dropna().copy()

    ###### ARIMA #########
    endog = sel_ind_ticker.copy()
    exog = sel_ind_nlargest_tickers.copy()
    p, q = 2, 1
    # get index lead returns for prediction
    df_rets[f'{endog[0]}_lead'] = df_rets[endog[0]].shift(1)
    df_rets.dropna(inplace=True)
    # get arima output
    st.write(df_rets.head())
    p, q, d, ma_resid, arima_params = get_ARMA_test(p, q, df_rets, endog, exog)

    ####### Kalman Filter #####
    xdim = p + d + q
    zdim = xdim
    # set up filter
    T, Q, Z, H, x0, P0, zs, state_vars, zs_index = set_up_kalman_filter(p, q, d, xdim, zdim, df_rets, ma_resid,
                                                              arima_params, endog, exog)
    # run filter
    X_out, P_out, X_pred, P_pred, LL_out = kalman_filter(xdim, zdim, p, q, d, x0, P0, zs, T, Q, Z, H, state_vars)

    df_xtrue = df_rets[endog].loc[zs_index].copy()
    ind = pd.DatetimeIndex([str(item) for item in zs_index])
    df_xtrue = pd.DataFrame(df_xtrue.values, index=ind, columns=endog)

    ind = pd.DatetimeIndex([*chain([str(item) for item in zs_index], [str(datetime.now().date() + timedelta(days=1))])])
    df_xpred = pd.DataFrame(X_pred[:, 0], index=ind, columns=[f'{endog[0]}_pred'])

    # get performance scoring
    conf_mat = pd.DataFrame(confusion_matrix(y_true=(df_xtrue >= 0),
                                             y_pred=(df_xpred.iloc[:-1] >= 0)),
                            index=['tn', 'fp'], columns=['fn', 'tp'])
    roc_score = roc_auc_score(y_true=(df_xtrue >= 0), y_score=(df_xpred.iloc[:-1] >= 0))

    #### Plotting #####
    fig1 = px.line(df_prices[sel_ind_ticker].values)

    # fill streamlit page
    st.plotly_chart(fig1, theme='streamlit', use_container_width=True)

    ##### Part 2 ######
    st.header('Forecasting output')
    tab1, tab2 = st.tabs(["Kalman Filter", "ARIMA"])
    # Kalman Filter
    with tab1:
        # streamlit cols
        b1, b2 = st.columns(2)

        b1.line_chart(pd.concat([df_xtrue, df_xpred], axis=1))
        b1.write(f'Tomorrows return is predicted to be: {round(df_xpred.iloc[-1].values[0], 3)}')

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(conf_mat, ax=ax, annot=True, cmap='winter')
        ax.set_title("Confusion Matrix")
        b2.write(fig)
        b2.write(f'Kalman Filter has ROC of {round(roc_score, 3)}')
        if roc_score < .5:
            b2.write(f'WARNING: This model has no predictive power')

    # ARIMA
    with tab2:
        st.plotly_chart(fig1, theme='streamlit', use_container_width=True)

    # Reset cache
    if clear_cache:
        st.cache_data.clear()