import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
# test2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score

from itertools import chain
from datetime import datetime, timedelta, date

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
    index_tickers = ['^STOXX50E', '^FTSE', '^OMX', '^FCHI', '^GDAXI', '^MDAXI', '^TECDAX', '^IBEX', '^GSPC', '^DJI',
                     '^AEX', '^IXIC']
    all_index_dict = dict(zip(all_indices, index_tickers))

    # streamlit side bar
    with st.sidebar:
        # select index
        SEL_IND = st.selectbox('What index to analyse?', tuple(all_indices))  # str
        SEL_IND_TICKER = [all_index_dict[SEL_IND]]  # list

        pull_data_start = st.date_input("Choose a start data for the following analysis", date(2017, 5, 1))

        #### Load Data #####
        pull_data_start = str(pull_data_start)
        pull_data_end = str(datetime.now().date())

        DF_PRICES, DF_RETS, SEL_IND_NLARGEST_TICKERS, LEAD_NAME = load_data(SEL_IND, SEL_IND_TICKER,
                                                                            pull_data_start, pull_data_end)

        # clear cache button
        clear_cache = st.button('Clear cache')

    ##### Section 1 #####
    st.header(f'{SEL_IND} price overview')
    st.write(
        f"""
            The past daily prices from {pull_data_start} to {pull_data_end}
            """
    )
    #### Plotting #####
    fig1 = px.line(DF_PRICES.reset_index(), y=SEL_IND_TICKER[0], x='index')
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
        str_start, str_end = str(dt_start), str(dt_end)
        st.write(f'The analysis will run for {analysis_time} weeks from {str_start} to {str_end}')

        # set chosen observation time
        df_prices_sel = DF_PRICES.loc[str_start:str_end]
        df_rets_sel = DF_RETS[str_start: str_end]

        ###### ARIMA #########
        endog = [LEAD_NAME]
        exog = SEL_IND_NLARGEST_TICKERS.copy()
        p, q = 3, 1

        # get arima output
        p, q, d, ma_resid, arima_params = get_ARMA_test(p, q, df_rets_sel, endog, exog)

        ####### Kalman Filter #####
        xdim = p + d + q
        zdim = xdim
        # set up filter
        T, Q, Z, H, x0, P0, zs, state_vars, zs_index = set_up_kalman_filter(p, q, d, xdim, zdim, df_rets_sel, ma_resid,
                                                                            arima_params, endog, exog,
                                                                            measurement_noise)
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
                                index=['negative', 'positive'], columns=['true', 'false'])
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
        c1, c2, c3 = st.columns([1, 1, 1])
        d1, d2 = st.columns([1, 1])

        hmm_states = c1.select_slider("How HMM states", range(1, 6), value=3)
        cv_samples = c2.select_slider("How many cross validation samples", range(1_000, 51_000, 1_000), value=2_000)
        hmm_init = c3.select_slider("HMM start init", range(10, 110, 10), value=20)

        # get data for CV
        data = DF_RETS.drop([item for item in DF_RETS.columns if SEL_IND_TICKER[0] in item], axis=1).copy()
        data = data.join(DF_RETS[[SEL_IND_TICKER[0], LEAD_NAME]])
        data = data.join(DF_PRICES[[f'{SEL_IND_TICKER[0]}_{item}' for item in ['High', 'Low', 'Open', 'Volume']]])
        data = data.join(DF_PRICES[SEL_IND_TICKER].rename(columns={SEL_IND_TICKER[0]: f'{SEL_IND_TICKER[0]}_price'}))

        # outlier selection
        mask = is_outlier(data[LEAD_NAME])
        data = data[~mask]
        train, test = train_test_split(data, test_size_split=[.8])

        # get cross validation and testing data
        arr_test, test_cols = get_hmm_features(arr=test.values, ind_ticker=SEL_IND_TICKER[0], lead_var=LEAD_NAME,
                                               cols_list=list(test.columns),
                                               n_largest_stocks=list(SEL_IND_NLARGEST_TICKERS))
        arr_train, train_cols = get_hmm_features(arr=train.values, ind_ticker=SEL_IND_TICKER[0], lead_var=LEAD_NAME,
                                                 cols_list=list(train.columns),
                                                 n_largest_stocks=list(SEL_IND_NLARGEST_TICKERS))
        arr_test = np.array(arr_test, dtype=float)
        arr_train = np.array(arr_train, dtype=float)
        arr_train_cv, train_cols_cv = get_CV_data(data_arr=train.values, cols_list=list(train.columns),
                                                  ind_ticker=SEL_IND_TICKER[0], lead_var=LEAD_NAME,
                                                  n_largest_stocks=list(SEL_IND_NLARGEST_TICKERS),
                                                  n_iterations=cv_samples)
        # scale data
        arr_test = arr_test.transpose()
        arr_train = arr_train.transpose()
        arr_train_cv = np.concatenate(arr_train_cv, axis=1).transpose()

        arr_train_cv = np.delete(arr_train_cv, get_index('id', train_cols_cv), axis=1)
        train_cols_cv.remove('id')

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
        states, statesg = get_hidden_states(train_cv_states, arr_train_cv[:, get_index('forecast_variable',
                                                                                       train_cols_cv, True)])
        st.write(statesg)

        fig = plt.figure()
        for i in set(states['states']):
            plt.hist(states[states['states'] == i]['rets'], bins='fd', alpha=.6, label=i)
        plt.legend()
        d1.write(fig)

        fig = plt.figure()
        sns.violinplot(states, x='states', y='rets')
        plt.plot([-1, 0, 1, 2, 3], [0, 0, 0, 0, 0], color='black')
        d2.write(fig)

        # fig = px.violin(states, x='states', y='rets')
        # d3.write(fig)

        test_states = mod.predict(X_test)
        X_test_df = pd.DataFrame(X_test, columns=train_cols_cv[:-1])
        X_test_df[f'{SEL_IND_TICKER[0]}_price'] = test[f'{SEL_IND_TICKER[0]}_price'].iloc[1:].values
        X_test_df[f'{SEL_IND_TICKER[0]}'] = test[f'{SEL_IND_TICKER[0]}'].iloc[1:].values
        X_test_df['date'] = list(test.index)[1:]
        X_test_df['is_test'] = [True] * len(X_test_df)

        train_states = mod.predict(X_train)
        X_train_df = pd.DataFrame(X_train, columns=train_cols[:-1])
        X_train_df[f'{SEL_IND_TICKER[0]}_price'] = train[f'{SEL_IND_TICKER[0]}_price'].iloc[1:].values
        X_train_df[f'{SEL_IND_TICKER[0]}'] = train[f'{SEL_IND_TICKER[0]}'].iloc[1:].values
        X_train_df['date'] = list(train.index)[1:]
        X_train_df['is_test'] = [False] * len(X_train_df)

        fig = plot_hmm_states(pd.concat([X_train_df, X_test_df], axis=0),
                              np.concatenate([train_states, test_states], axis=0),
                              f'{SEL_IND_TICKER[0]}_price', f'{SEL_IND_TICKER[0]}', 'date', 'is_test')
        st.write('Out of sample test')
        st.write(fig)

    # Reset cache
    if clear_cache:
        st.cache_data.clear()
