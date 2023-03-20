import pandas as pd
import streamlit as st
import plotly.express as px

import matplotlib.pyplot as plt

import os

from settings import DATA_DIR

if __name__ == '__main__':
    # streamlit setup
    st.set_page_config(page_title='A binary guide to the S&P 500', layout='wide')

    # Part 1
    st.header('SP 500 price overview')
    a1, a2 = st.columns([1, 1])
    b1, b2, b3 = st.columns([1, 2, 1])

    # Part 2
    st.header('Risk Portfolio Allocation')
    d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
    e1, e2 = st.columns([1, 1])

    # Load data
    sp500_ticker = ['^GSPC']
    start = '2020-01-01'
    end = '2022-12-31'

    df_rets = pd.read_csv(os.path.join(DATA_DIR, 'returns.csv'), index_col='date').drop('Unnamed: 0', axis=1)
    fig1 = px.line(df_rets[sp500_ticker])
    fig2 = px.histogram(df_rets[sp500_ticker])

    tab1, tab2 = st.tabs(["Line plot", "Histogram"])
    with tab1:
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig2, theme=None, use_container_width=True)

    # st.plotly_chart(fig1, title=f'{sp500_ticker[0]} returns')
    # st.ploty_chart(fig2, title=f'{sp500_ticker[0]} returns histogram')