from src.toml import load_toml
from src.hmm import plot_hmm_states

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import streamlit as st
from datetime import datetime, timedelta

import os

import plotly.express as px

from settings import PROJECT_ROOT

global SECTION_1, SECTION_2, SECTION_3, SEL_IND, SEL_IND_TICKER, PULL_START_DATE, PULL_END_DATE, TAB1, TAB2, TAB3

config = load_toml(os.path.join(PROJECT_ROOT, 'config.toml'))
config_all_index_dict = {y: x for y, x in list(config['indices'].values())}
config_all_index = [item[0] for item in list(config['indices'].values())]
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


def set_up_page():
    global SECTION_1, SECTION_2, SECTION_3, SEL_IND, SEL_IND_TICKER, PULL_START_DATE, PULL_END_DATE, TAB1, TAB2, TAB3

    st.set_page_config(page_title=config['page_set_up']['title'], layout='wide')
    SECTION_1 = st.container()
    SECTION_2 = st.container()
    SECTION_3 = st.container()

    with st.sidebar:
        # select index
        SEL_IND = st.selectbox(config['page_set_up']['select_ind_text'], tuple(config_all_index))  # str
        SEL_IND_TICKER = [config_all_index_dict[SEL_IND]]  # list

        # select start data
        PULL_START_DATE = st.date_input(config['page_set_up']['select_start_text'],
                                        datetime(*default_pull_start_date))
        PULL_END_DATE = datetime.now().date()

        # clear cache button
        clear_cache = st.button(config['page_set_up']['cache_button'])
        if clear_cache:
            st.cache_data.clear()

    SECTION_1.header(f"{SEL_IND} {config['page_set_up']['section_1_header']}")
    SECTION_1.write(f"""The past daily prices from {str(PULL_START_DATE)} to {PULL_END_DATE}""")

    SECTION_2.header(config['page_set_up']['section_2_header'])
    TAB1, TAB2, TAB3 = SECTION_2.tabs(config['page_set_up']['tab_options'])

    return SEL_IND, SEL_IND_TICKER, PULL_START_DATE, PULL_END_DATE


def set_up_sliders():
    c1, c2, c3 = TAB1.columns([1, 1, 1])
    measurement_noise = c1.select_slider(config['streamlit_sliders_text']['kf_measurement_noise'],
                                         options=np.arange(*slider_kf_measurement),
                                         value=default_KF_measurement_noise)
    analysis_time = c3.select_slider(config['streamlit_sliders_text']['kf_analysis_time'],
                                     options=range(*slider_kf_analysis_time),
                                     value=default_KF_analysis_time)
    cv_samples_kalman = c2.select_slider(config['streamlit_sliders_text']['kf_cv_samples'],
                                         options=range(*slider_kf_cv_samples),
                                         value=default_KF_cv_samples)
    test_sample_start = PULL_END_DATE - timedelta(weeks=analysis_time)
    TAB1.write(f'The analysis will run for {analysis_time} weeks from {str(test_sample_start)} to {str(PULL_END_DATE)}')

    TAB2.write('ARIMA')

    TAB3.write('HMM model')
    c1, c2, c3 = TAB3.columns([1, 1, 1])
    hmm_states = c1.select_slider(config['streamlit_sliders_text']['hmm_no_states'],
                                  options=range(*slider_hmm_no_states), value=default_HMM_no_states)
    cv_samples = c2.select_slider(config['streamlit_sliders_text']['hmm_cv_samples'],
                                  options=range(*slider_hmm_cv_samples), value=default_HMM_cv_samples)
    hmm_init = c3.select_slider(config['streamlit_sliders_text']['hmm_start_init'],
                                options=range(*slider_hmm_start_init), value=default_HMM_start_init)

    return hmm_states, cv_samples, hmm_init, test_sample_start, measurement_noise, analysis_time, cv_samples_kalman


def st_plot_output(df_prices, kf_xtrue, kf_xpred, kf_xfilt, kf_conf_mat, kf_roc_score,
                   hmm_cv_statesg, hmm_cv_states, hmm_xtrain, hmm_xtest, hmm_train_states, hmm_test_states,
                   arma_true, arma_pred):

    fig1 = px.line(df_prices.reset_index(), y=SEL_IND_TICKER[0], x='index')
    SECTION_1.plotly_chart(fig1, theme='streamlit', use_container_width=True)

    # kalman filter
    b1, b2 = TAB1.columns([3, 1])
    b1.write('Returns chart')
    b1.line_chart(pd.concat([kf_xtrue, kf_xpred, kf_xfilt], axis=1))
    b1.write(f'Tomorrows return is predicted to be: {round(kf_xpred.iloc[-1].values[0], 3)}')

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(kf_conf_mat, ax=ax, annot=True, cmap='winter')
    ax.set_title("Confusion Matrix")
    b2.write(fig)

    b2.write(f'Kalman Filter has ROC of {round(kf_roc_score, 3)}')
    if kf_roc_score < .5:
        b2.write(f'WARNING: This model has no predictive power')

    # arima
    TAB2.line_chart(pd.concat([pd.Series(arma_true, name='ARMA_true'),
                               pd.Series(arma_pred , name='ARMA_pred')], axis=1))
    # fig, ax = plt.subplots(figsize=(15, 5))
    # ax.plot(arma_true)
    # ax.plot(arma_pred)
    # TAB2.plotly_chart(fig, theme='streamlit', use_container_width=True)

    # hmm
    d1, d2 = TAB3.columns([1, 1])
    TAB3.dataframe(hmm_cv_statesg)

    fig = plt.figure()
    for i in set(hmm_cv_states['states']):
        plt.hist(hmm_cv_states[hmm_cv_states['states'] == i]['rets'], bins='fd', alpha=.6, label=i)
    plt.legend()
    d1.write(fig)

    fig = plt.figure()
    sns.violinplot(hmm_cv_states, x='states', y='rets')
    plt.plot([-1, 0, 1, 2, 3], [0, 0, 0, 0, 0], color='black')
    plt.tight_layout()
    d2.write(fig)

    fig = plot_hmm_states(pd.concat([hmm_xtrain, hmm_xtest], axis=0),
                          np.concatenate([hmm_train_states, hmm_test_states], axis=0),
                          f'{SEL_IND_TICKER[0]}_price', f'{SEL_IND_TICKER[0]}', 'date', len(hmm_xtrain))
    plt.tight_layout()
    TAB3.write('Out of sample test')
    TAB3.write(fig)

    pass
