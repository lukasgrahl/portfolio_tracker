import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score

from itertools import chain
from datetime import datetime, timedelta

from src.pull_data import load_data
from src.utils import train_test_split, is_outlier, get_index
from src.filter import get_ARMA_test, set_up_kalman_filter, kalman_filter
from src.hmm import get_hmm, get_hmm_features, get_CV_data, get_hidden_states, plot_hmm_states
from sklearn.preprocessing import scale

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
        sel_ind = st.selectbox('What index to analyse?', tuple(all_indices)) # str
        sel_ind_ticker = [all_index_dict[sel_ind]] # list

        #### Load Data #####
        pull_data_start = "2000-01-01"
        pull_data_end = str(datetime.now().date())

        df_prices, df_rets, sel_ind_nlargest_tickers = load_data(sel_ind, sel_ind_ticker, pull_data_start, pull_data_end)

        # clear cache button
        clear_cache = st.button('Clear cache')

    ##### Section 1 #####
    st.header(f'{sel_ind} price overview')
    st.write(
        f"""
            The past daily prices from {pull_data_start} to {pull_data_end}
            """
    )
    #### Plotting #####
    fig1 = px.line(df_prices[sel_ind_ticker].values)
    st.plotly_chart(fig1, theme='streamlit', use_container_width=True)

    ##### Section 2 ######
    st.header('Forecasting output')
    tab1, tab2, tab3 = st.tabs(["Kalman Filter", "ARIMA", "HMM"])

    # Kalman filter
    with tab1:
        c1, c2 = st.columns([1, 1])
        measurement_noise = c2.select_slider('Kalman Filter measurment noise', options=np.arange(0, 2.1, .1), value=.1)
        # select time window
        analysis_time = c1.select_slider('How many weeks would you like the analysis to run on?',
                                         options=list(range(10, 260, 10)),
                                         value=20)
        dt_end = datetime.now().date()
        dt_start = dt_end - timedelta(weeks=analysis_time)
        start = str(dt_start)
        end = str(dt_end)
        st.write(f'The analysis will run for {analysis_time} weeks from {start} to {end}')

        # set chosen observation time
        df_prices_sel = df_prices.loc[start:end]
        df_rets_sel = np.log(df_prices_sel / df_prices_sel.shift(1)).dropna().copy()

        ###### ARIMA #########
        endog = [f'{sel_ind_ticker[0]}_lead']
        exog = sel_ind_nlargest_tickers.copy()
        p, q = 3, 1

        # get index lead returns for prediction
        df_rets_sel[endog[0]] = df_rets_sel[sel_ind_ticker[0]].shift(-1)
        df_rets_sel.dropna(inplace=True)
        # get arima output
        p, q, d, ma_resid, arima_params = get_ARMA_test(p, q, df_rets_sel, endog, exog)


        ####### Kalman Filter #####
        xdim = p + d + q
        zdim = xdim
        # set up filter
        T, Q, Z, H, x0, P0, zs, state_vars, zs_index = set_up_kalman_filter(p, q, d, xdim, zdim, df_rets_sel, ma_resid,
                                                                            arima_params, endog, exog, measurement_noise)
        # run filter
        X_out, P_out, X_pred, P_pred, LL_out = kalman_filter(xdim, zdim, p, q, d, x0, P0, zs, T, Q, Z, H, state_vars)

        # get output as pd.DataFrame
        df_xtrue = df_rets_sel[endog].loc[zs_index].copy()
        ind = pd.DatetimeIndex([str(item) for item in zs_index])
        df_xtrue = pd.DataFrame(df_xtrue.values, index=ind, columns=endog)
        df_xfilt = pd.DataFrame(X_out[:, 0], index=ind, columns=[f'{endog[0]}_filter'])

        ind = pd.DatetimeIndex([*chain(
            [str(item) for item in zs_index],
            [str(datetime.now().date() + timedelta(days=1))]
        )])
        df_xpred = pd.DataFrame(X_pred[:, 0], index=ind, columns=[f'{endog[0]}_pred'])

        # get performance scoring
        conf_mat = pd.DataFrame(confusion_matrix(y_true=(df_xtrue >= 0),
                                                 y_pred=(df_xpred.iloc[:-1] >= 0)),
                                index=['tn', 'fp'], columns=['fn', 'tp'])
        conf_mat = conf_mat / len(df_xtrue)
        roc_score = roc_auc_score(y_true=(df_xtrue >= 0), y_score=(df_xpred.iloc[:-1] >= 0))

        # Streamlit
        b1, b2 = st.columns([3, 1])

        b1.write('Returns chart')
        b1.line_chart(pd.concat([df_xtrue, df_xpred, df_xfilt], axis=1))
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

    # HMM
    with tab3:
        st.write('HMM model')
        c1, c2 = st.columns([1, 1])

        # get data for CV
        data = df_rets.drop([item for item in df_rets.columns if sel_ind_ticker[0] in item], axis=1).copy()
        data = data.join(df_rets[sel_ind_ticker[0]])
        data = data.join(df_prices[[f'{sel_ind_ticker[0]}_{item}' for item in ['High', 'Low', 'Open', 'Volume']]])
        data = data.join(df_prices[sel_ind_ticker].rename(columns={sel_ind_ticker[0]: f'{sel_ind_ticker[0]}_price'}))

        # outlier selection
        mask = is_outlier(data[sel_ind_ticker[0]])
        data = data[~mask]
        train, test = train_test_split(data, test_size_split=[.8])



        # get cross validation and testing data
        arr_test, test_cols = get_hmm_features(test.values, sel_ind_ticker[0],
                                               list(test.columns), list(sel_ind_nlargest_tickers))
        arr_test = np.array(arr_test, dtype=float)
        cv_train, train_cols = get_CV_data(train.values, list(train.columns), sel_ind_ticker[0],
                                           n_largest_stocks=list(sel_ind_nlargest_tickers), n_iterations=5_000)
        # scale data
        arr_test = arr_test.transpose()
        arr_train = np.concatenate(cv_train, axis=1).transpose()

        arr_train = np.delete(arr_train, get_index('id', train_cols), axis=1)
        train_cols.remove('id')

        arr_train = np.column_stack([scale(arr_train[:, i]) for i in range(arr_train.shape[1])])
        arr_test = np.column_stack([scale(arr_test[:, i]) for i in range(arr_test.shape[1])])

        # get train and test sets
        X_test = arr_test[:, ~get_index('forecast_variable', test_cols, True)].copy()
        y_test = arr_test[:, get_index('forecast_variable', test_cols, True)].copy()
        X_train = arr_train[:, ~get_index('forecast_variable', train_cols, True)].copy()
        y_train = arr_train[:, get_index('forecast_variable', train_cols, True)].copy()


        # train model
        mod, hidden_states = get_hmm(X_train, y_train, n_components=3, n_int=50)
        states, statesg = get_hidden_states(hidden_states, y_train)

        st.write(statesg)

        fig = plt.figure()
        for i in set(states['states']):
            plt.hist(states[states['states'] == i]['rets'], bins='fd', alpha=.6, label=i)
        plt.legend()
        c1.write(fig)

        fig = plt.figure()
        sns.violinplot(states, x='states', y='rets')
        plt.plot([-1, 0, 1, 2, 3], [0, 0, 0, 0, 0], color='black')
        c2.write(fig)

        hidden_states = mod.predict(X_test)
        X_test = pd.DataFrame(X_test, columns=train_cols[:-1])
        X_test[f'{sel_ind_ticker[0]}_price'] = test[f'{sel_ind_ticker[0]}_price'].iloc[1:].values
        X_test[f'{sel_ind_ticker[0]}'] = test[f'{sel_ind_ticker[0]}'].iloc[1:].values
        X_test['date'] = list(test.index)[1:]
        fig = plot_hmm_states(X_test, hidden_states, f'{sel_ind_ticker[0]}_price', f'{sel_ind_ticker[0]}', 'date')
        st.write('Out of sample test')
        st.write(fig)

    # Reset cache
    if clear_cache:
        st.cache_data.clear()