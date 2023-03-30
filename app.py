import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date

from src.pull_data import load_data
from src.utils import get_binary_metric, is_outlier
from src.filter import run_kalman_filter
from src.hmm import run_hmm, plot_hmm_states

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

        PULL_START_DATE = st.date_input("Choose a start data for the following analysis", date(2017, 5, 1))

        #### Load Data #####
        PULL_START_DATE_STR = str(PULL_START_DATE)
        pull_data_end = str(datetime.now().date())

        DF_PRICES, DF_RETS, SEL_IND_NLARGEST_TICKERS, LEAD_NAME = load_data(SEL_IND, SEL_IND_TICKER,
                                                                            PULL_START_DATE_STR, pull_data_end)

        # clear cache button
        clear_cache = st.button('Clear cache')

    ##### Section 1 #####
    st.header(f'{SEL_IND} price overview')
    st.write(
        f"""
            The past daily prices from {PULL_START_DATE_STR} to {pull_data_end}
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
        c1, c2, c3 = st.columns([1, 1, 1])
        measurement_noise = c1.select_slider('Kalman Filter measurment noise', options=np.arange(0, 2.1, .1), value=.1)
        # select time window
        analysis_time = c3.select_slider('How many weeks would you like the analysis to run on?',
                                         options=list(range(10, 260, 10)), value=20)
        cv_samples_kalman = c2.select_slider('Cross validation samples', options=range(10, 80, 10), value=20)
        dt_end = datetime.now().date()
        dt_start = dt_end - timedelta(weeks=analysis_time)

        str_start, str_end = str(dt_start), str(dt_end)
        st.write(f'The analysis will run for {analysis_time} weeks from {str_start} to {str_end}')

        endog = [LEAD_NAME]
        exog = SEL_IND_NLARGEST_TICKERS.copy()

        # run cross validation
        cv_output = []
        cv_index_len = DF_PRICES.loc[PULL_START_DATE: dt_start].shape[0] # index length
        from random import randint
        for i in range(0, cv_samples_kalman):
            cv_start = randint(0, (cv_index_len - analysis_time * 5)) # take
            cv_end = cv_start + (analysis_time * 5) #

            df_prices_sel = DF_PRICES.iloc[cv_start: cv_end].copy()
            df_rets_sel = DF_RETS.iloc[cv_start: cv_end].copy()

            df_xtrue, df_xpred, df_xfilt = run_kalman_filter(endog, exog, df_rets_sel, measurement_noise)
            cv_output.append([df_xtrue.values.reshape(-1), df_xpred.iloc[:-1].values.reshape(-1)])

        cv_output = np.array(cv_output)
        conf_mat, roc_score = get_binary_metric(np.concatenate(cv_output[:, 0, :]), np.concatenate(cv_output[:, 1, :]), cut_off=0)

        # set chosen observation time
        df_prices_sel = DF_PRICES.loc[str_start:str_end].copy()
        df_rets_sel = DF_RETS.loc[str_start: str_end].copy()

        df_xtrue, df_xpred, df_xfilt = run_kalman_filter(endog, exog, df_rets_sel, measurement_noise)

        # get performance scoring
        # conf_mat, roc_score = get_binary_metric(df_xtrue, df_xpred[:-1], cut_off=0.)

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
        cv_samples = c2.select_slider("How many cross validation samples", range(1_000, 51_000, 1_000), value=1_000)
        hmm_init = c3.select_slider("HMM start init", range(10, 110, 10), value=20)

        # get data for CV
        data = DF_RETS.drop([item for item in DF_RETS.columns if SEL_IND_TICKER[0] in item], axis=1).copy()
        data = data.join(DF_RETS[[SEL_IND_TICKER[0], LEAD_NAME]])
        data = data.join(DF_PRICES[[f'{SEL_IND_TICKER[0]}_{item}' for item in ['High', 'Low', 'Open', 'Volume']]])
        data = data.join(DF_PRICES[SEL_IND_TICKER].rename(columns={SEL_IND_TICKER[0]: f'{SEL_IND_TICKER[0]}_price'}))

        # outlier selection
        mask = is_outlier(data[LEAD_NAME])
        data = data[~mask]

        # run hmm
        run_out = run_hmm(data, SEL_IND_TICKER, LEAD_NAME, SEL_IND_NLARGEST_TICKERS, hmm_states, hmm_init, cv_samples)
        # get hmm output
        mod, train_cv_states, cv_states, cv_statesg, test, train, train_cv = run_out
        X_test, y_test, X_test_df, test_states = test
        X_train, y_train, X_train_df, train_states = train
        X_train_cv, y_train_cv = train_cv

        # overview on states returns distribution
        st.write(cv_statesg)

        fig = plt.figure()
        for i in set(cv_states['states']):
            plt.hist(cv_states[cv_states['states'] == i]['rets'], bins='fd', alpha=.6, label=i)
        plt.legend()
        d1.write(fig)

        fig = plt.figure()
        sns.violinplot(cv_states, x='states', y='rets')
        plt.plot([-1, 0, 1, 2, 3], [0, 0, 0, 0, 0], color='black')
        plt.tight_layout()
        d2.write(fig)

        # fig = px.violin(states, x='states', y='rets')
        # d3.write(fig)

        # plot train test
        fig = plot_hmm_states(pd.concat([X_train_df, X_test_df], axis=0),
                              np.concatenate([train_states, test_states], axis=0),
                              f'{SEL_IND_TICKER[0]}_price', f'{SEL_IND_TICKER[0]}', 'date', len(X_train_df))
        plt.tight_layout()
        st.write('Out of sample test')
        st.write(fig)

    # Reset cache
    if clear_cache:
        st.cache_data.clear()
