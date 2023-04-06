import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from datetime import datetime, timedelta, date

from src.pull_data import load_data
from src.utils import get_binary_metric, is_outlier
from src.filter import run_kalman_filter, get_kalman_cv
from src.hmm import run_hmm, plot_hmm_states
from src.get_toml import get_toml_data
import os

from settings import PROJECT_ROOT

if __name__ == '__main__':

    # get config globals
    config = get_toml_data(os.path.join(PROJECT_ROOT, 'config.toml'))
    config_all_index_dict = {y: x for y, x in list(config['indices'].values())}
    config_all_index = [item[0] for item in list(config['indices'].values())]

    # get data related vals
    train_test_size = config['data']['train_test_size']
    outlier_interval = config['data']['outlier_std_interval']

    # get slier values
    slider_kf_measurement = config['streamlit_sliders']['kf_measurement_noise']
    slider_kf_analysis_time = config['streamlit_sliders']['kf_analysis_time']
    slider_kf_cv_samples = config['streamlit_sliders']['kf_cv_samples']
    slider_hmm_no_states = config['streamlit_sliders']['hmm_no_states']
    slider_hmm_cv_samples = config['streamlit_sliders']['hmm_cv_samples']
    slider_hmm_start_init = config['streamlit_sliders']['hmm_start_init']

    # set default values for sliders
    default_pull_start_date = config['default_values']['pull_start_date']
    default_KF_cv_samples = config['default_values']['kf_cv_samples']
    default_KF_analysis_time = config['default_values']['kf_analysis_time']
    default_KF_measurement_noise = config['default_values']['kf_measurement_noise']
    default_HMM_no_states = config['default_values']['hmm_no_states']
    default_HMM_cv_samples = config['default_values']['hmm_cv_samples']
    default_HMM_start_init = config['default_values']['hmm_start_init']
    default_HMM_cv_sample_sizes = config['default_values']['hmm_cv_sample_sizes']

    # streamlit sidebar
    st.set_page_config(page_title='A binary guide to the S&P 500', layout='wide')
    with st.sidebar:
        # select index
        SEL_IND = st.selectbox('What index to analyse?', tuple(config_all_index))  # str
        SEL_IND_TICKER = [config_all_index_dict[SEL_IND]]  # list

        # select start data
        PULL_START_DATE = st.date_input("Choose a start data for the following analysis",
                                        datetime(*default_pull_start_date)) # value=default_pull_start_date)
        PULL_END_DATE = datetime.now().date()

        # clear cache button
        clear_cache = st.button('Clear cache')
        if clear_cache:
            st.cache_data.clear()

    ##### Section 1 #####
    st.header(f'{SEL_IND} price overview')
    st.write(f"""The past daily prices from {str(PULL_START_DATE)} to {PULL_END_DATE}""")

    #### Load Data #####
    DF_PRICES, DF_RETS, SEL_IND_NLARGEST_TICKERS, LEAD_NAME = load_data(SEL_IND, SEL_IND_TICKER,
                                                                        str(PULL_START_DATE), str(PULL_END_DATE))

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
        measurement_noise = c1.select_slider('Kalman Filter measurment noise',
                                             options=np.arange(*slider_kf_measurement),
                                             value=default_KF_measurement_noise)
        analysis_time = c3.select_slider('How many weeks would you like the analysis to run on?',
                                         options=range(*slider_kf_analysis_time),
                                         value=default_KF_analysis_time)
        cv_samples_kalman = c2.select_slider('Cross validation samples',
                                             options=range(*slider_kf_cv_samples),
                                             value=default_KF_cv_samples)
        dt_start = PULL_END_DATE - timedelta(weeks=analysis_time)
        st.write(f'The analysis will run for {analysis_time} weeks from {str(dt_start)} to {str(PULL_END_DATE)}')

        # set kalman filter values

        endog = [LEAD_NAME]
        exog = SEL_IND_NLARGEST_TICKERS.copy()
        cv_index_len = DF_PRICES.loc[PULL_START_DATE: dt_start].shape[0]  # index length

        # cross validate kalman filter
        conf_mat, roc_score = get_kalman_cv(data=DF_RETS, endog=[LEAD_NAME], exog=SEL_IND_NLARGEST_TICKERS.copy(),
                                            measurement_noise=measurement_noise, cv_index_len=cv_index_len,
                                            sample_len_weeks=analysis_time, no_samples=cv_samples_kalman)

        # set chosen observation time
        df_prices_sel = DF_PRICES.loc[str(dt_start): str(PULL_END_DATE)].copy()
        df_rets_sel = DF_RETS.loc[str(dt_start): str(PULL_END_DATE)].copy()
        # run filter on test data
        df_xtrue, df_xpred, df_xfilt = run_kalman_filter(endog=[LEAD_NAME], exog=SEL_IND_NLARGEST_TICKERS.copy(),
                                                         data=df_rets_sel, measurement_noise=measurement_noise)

        # ST present output
        b1, b2 = st.columns([3, 1])
        b1.write('Returns chart')
        b1.line_chart(pd.concat([df_xtrue, df_xpred, df_xfilt], axis=1))
        b1.write(f'Tomorrows return is predicted to be: {round(df_xpred.iloc[-1].values[0], 3)}')

        # plot
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(conf_mat, ax=ax, annot=True, cmap='winter')
        ax.set_title("Confusion Matrix")

        b2.write(fig)
        b2.write(f'Kalman Filter has ROC of {round(roc_score, 3)}')
        if roc_score < .5:
            b2.write(f'WARNING: This model has no predictive power')

    # ARIMA
    with tab2:
        
    from arima_values import p, d, q
    
       # ST select ARIMA inputs
    c1, c2, c3 = st.columns([1, 1, 1])
    p = c1.slider('ARIMA(p)', 0, 10, value=p)
    d = c2.slider('ARIMA(d)', 0, 10, value=d)
    q = c3.slider('ARIMA(q)', 0, 10, value=q)

    order = c1.text_input('ARIMA order (p, d, q)', '1, 0, 1')
    analysis_time = c3.select_slider('How many weeks would you like the analysis to run on?',
                                 options=range(1, 53),
                                 value=26)
    cv_samples_arima = c2.select_slider('Cross validation samples',
                                     options=range(1, 11),
                                     value=5)
    dt_start = PULL_END_DATE - timedelta(weeks=analysis_time)
    st.write(f'The analysis will run for {analysis_time} weeks from {str(dt_start)} to {str(PULL_END_DATE)}')

    # set ARIMA values
    p, d, q = [int(x.strip()) for x in order.split(',')]
    mydata = pd.read_csv('mydata.csv')
    model = sm.tsa.arima.ARIMA(mydata.value, order=(p, d, q))

    # cross validate ARIMA model
    conf_mat, roc_score = get_arima_cv(data=mydata.value, model=model, cv_samples=cv_samples_arima,
                                   sample_len_weeks=analysis_time, start_date=PULL_START_DATE, end_date=dt_start)

    # fit ARIMA model to selected data
    modelfit = model.fit(start_params=None)
    A = modelfit.model.ssm.transition[..., 0]
    params = modelfit.params

    # ST present output
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(mydata.index, mydata.value, label='Original Data')
    ax1.plot(mydata.index, modelfit.fittedvalues, label='ARIMA Model')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.set_title('ARIMA Model vs. Original Data')
    ax1.legend()
    st.pyplot(fig1)

    b1, b2 = st.columns([3, 1])
    b1.write('ARIMA Model Summary:')
    b1.write(modelfit.summary().as_html())

    # plot
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sns.heatmap(conf_mat, ax=ax2, annot=True, cmap='winter')
    ax2.set_title("Confusion Matrix")

    b2.write(fig2)
    b2.write(f'ARIMA Model has ROC of {round(roc_score, 3)}')
    if roc_score < .5:
        b2.write(f'WARNING: This model has no predictive power') 
        
     #st.plotly_chart(fig1, theme='streamlit', use_container_width=True)

    
    # HMM
    with tab3:
        st.write('HMM model')
        c1, c2, c3 = st.columns([1, 1, 1])
        d1, d2 = st.columns([1, 1])

        hmm_states = c1.select_slider("How HMM states", options=range(*slider_hmm_no_states),
                                      value=default_HMM_no_states)
        cv_samples = c2.select_slider("How many cross validation samples", options=range(*slider_hmm_cv_samples),
                                      value=default_HMM_cv_samples)
        hmm_init = c3.select_slider("HMM start init", options=range(*slider_hmm_start_init),
                                    value=default_HMM_start_init)

        # get data for CV
        data = DF_RETS.drop([item for item in DF_RETS.columns if SEL_IND_TICKER[0] in item], axis=1).copy()
        data = data.join(DF_RETS[[SEL_IND_TICKER[0], LEAD_NAME]])
        data = data.join(DF_PRICES[[f'{SEL_IND_TICKER[0]}_{item}' for item in ['High', 'Low', 'Open', 'Volume']]])
        data = data.join(DF_PRICES[SEL_IND_TICKER].rename(columns={SEL_IND_TICKER[0]: f'{SEL_IND_TICKER[0]}_price'}))

        # outlier selection
        mask = is_outlier(data[LEAD_NAME], outlier_interval)
        data = data[~mask]

        # run hmm
        run_out = run_hmm(data=data, sel_ind_ticker=SEL_IND_TICKER, lead_name=LEAD_NAME,
                          sel_ind_nlargest_ticker=SEL_IND_NLARGEST_TICKERS, hmm_states=hmm_states, hmm_init=hmm_init,
                          cv_samples=cv_samples, train_test_size=train_test_size,
                          cv_sample_sizes=tuple(default_HMM_cv_sample_sizes))
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
