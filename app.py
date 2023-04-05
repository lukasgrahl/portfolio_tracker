import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date

from src.pull_data import load_data
from src.utils import get_binary_metric, is_outlier
from src.filter import run_kalman_filter, get_kalman_cv
from src.hmm import run_hmm, plot_hmm_states
from src.get_toml import get_toml_data
import os

from settings import DATA_DIR, PROJECT_ROOT

if __name__ == '__main__':
    config = get_toml_data(os.path.join(PROJECT_ROOT, 'config.toml'))
    all_index_dict = {y: x for y, x in list(config['indices'].values())}
    all_index = [item[0] for item in list(config['indices'].values())]

    st.set_page_config(page_title='A binary guide to the S&P 500', layout='wide')

    # streamlit sidebar
    with st.sidebar:
        # select index
        SEL_IND = st.selectbox('What index to analyse?', tuple(all_index))  # str
        SEL_IND_TICKER = [all_index_dict[SEL_IND]]  # list

        # select start data
        PULL_START_DATE = st.date_input("Choose a start data for the following analysis", date(2017, 5, 1))
        PULL_END_DATE = datetime.now().date()

        # clear cache button
        clear_cache = st.button('Clear cache')
        # Reset cache
        if clear_cache:
            st.cache_data.clear()

    ##### Section 1 #####
    st.header(f'{SEL_IND} price overview')
    st.write(f"""The past daily prices from {str(PULL_START_DATE)} to {PULL_END_DATE}""")

    #### Load Data #####
    DF_PRICES, DF_RETS, SEL_IND_NLARGEST_TICKERS, LEAD_NAME = load_data(SEL_IND, SEL_IND_TICKER,
                                                                        str(PULL_START_DATE), PULL_END_DATE)

    #### Plotting #####
    fig1 = px.line(DF_PRICES.reset_index(), y=SEL_IND_TICKER[0], x='index')
    st.plotly_chart(fig1, theme='streamlit', use_container_width=True)

    ##### Section 2 ######
    st.header('Forecasting output')
    tab1, tab2, tab3 = st.tabs(["Kalman Filter", "ARIMA", "HMM"])

    # Kalman filter
    with tab1:
        # ST select kalman filter inputs
        c1, c2, c3 = st.columns([1, 1, 1])
        measurement_noise = c1.select_slider('Kalman Filter measurment noise', options=np.arange(0, 2.1, .1), value=.1)
        analysis_time = c3.select_slider('How many weeks would you like the analysis to run on?',
                                         options=list(range(10, 260, 10)), value=20)
        cv_samples_kalman = c2.select_slider('Cross validation samples', options=range(10, 80, 10), value=20)
        dt_start = PULL_END_DATE - timedelta(weeks=analysis_time)
        st.write(f'The analysis will run for {analysis_time} weeks from {str(dt_start)} to {str(PULL_END_DATE)}')

        # set kalman filter values

        endog = [LEAD_NAME]
        exog = SEL_IND_NLARGEST_TICKERS.copy()
        cv_index_len = DF_PRICES.loc[PULL_START_DATE: dt_start].shape[0]  # index length

        # cross validate kalman filter
        conf_mat, roc_score = get_kalman_cv(DF_RETS, endog, exog, measurement_noise, cv_index_len,
                                            analysis_time, cv_samples_kalman)

        # set chosen observation time
        df_prices_sel = DF_PRICES.loc[str(dt_start): str(PULL_END_DATE)].copy()
        df_rets_sel = DF_RETS.loc[str(dt_start): str(PULL_END_DATE)].copy()
        # run filter on test data
        df_xtrue, df_xpred, df_xfilt = run_kalman_filter(endog, exog, df_rets_sel, measurement_noise)

        # ST present output
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

        hmm_states = c1.select_slider("How HMM states", range(1, 6), value=2)
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

        # plot train test
        fig = plot_hmm_states(pd.concat([X_train_df, X_test_df], axis=0),
                              np.concatenate([train_states, test_states], axis=0),
                              f'{SEL_IND_TICKER[0]}_price', f'{SEL_IND_TICKER[0]}', 'date', len(X_train_df))
        plt.tight_layout()
        st.write('Out of sample test')
        st.write(fig)


