import pandas as pd
import numpy as np
from src.pull_data import get_index_nlargest_composits, get_yf_ticker_data
from itertools import chain
from src.utils import train_test_split
from sklearn.preprocessing import scale
from src.hmm import get_index
import matplotlib.pyplot as plt
from src.hmm import get_hmm, get_hidden_states, get_hmm_features, plot_hmm_states, get_CV_data
import seaborn as sns

if __name__ == '__main__':
    # user input
    start = '2015-01-01'
    end = '2022-12-31'
    sel_ind = 'OMX Stockholm 30'
    sel_ind_ticker = ['^OMX']

    # pull data
    sel_ind_composit_tickers, _, sel_ind_nlargest_tickers, success = get_index_nlargest_composits(sel_ind, n=5)
    # if success <= .8: st.write(f'Market cap was only available for {success*100: .1f} %  of composits')
    df_prices = get_yf_ticker_data(sel_ind_nlargest_tickers, start, end)
    df_prices = df_prices.join(get_yf_ticker_data(sel_ind_ticker, start, end, price_kind=['Open', 'High', 'Low', 'Volume', 'Adj Close']))
    df_prices.columns = [item if '_Adj Close' not in item else item[:-10] for item in df_prices.columns]
    # df_prices = get_yf_ticker_data([*chain(sel_ind_ticker, sel_ind_nlargest_tickers)], start, end)

    # replace zero in volume cols
    x = df_prices[df_prices[f'{sel_ind_ticker[0]}_Volume'] == 0].index
    df_prices.loc[x, f'{sel_ind_ticker[0]}_Volume'] = [1] * len(x)

    # get log return data
    df_rets = np.log(df_prices / df_prices.shift(1)).dropna().copy()

    # get data for CV
    data = df_rets.drop([item for item in df_rets.columns if sel_ind_ticker[0] in item], axis=1).copy()
    data = data.join(df_rets[sel_ind_ticker[0]])
    data = data.join(df_prices[[f'{sel_ind_ticker[0]}_{item}' for item in ['High', 'Low', 'Open', 'Volume']]])
    data = data.join(df_prices[sel_ind_ticker].rename(columns={sel_ind_ticker[0]: f'{sel_ind_ticker[0]}_price'}))
    train, test = train_test_split(data, test_size_split=[.8])

    # get cross validation and testing data
    arr_test, _ = get_hmm_features(test.values, sel_ind_ticker[0],
                                   list(test.columns), list(sel_ind_nlargest_tickers))
    arr_test = np.array(arr_test, dtype=float)
    cv_train, cv_cols = get_CV_data(train.values, list(train.columns), sel_ind_ticker[0],
                                    exogenous_vars=list(sel_ind_nlargest_tickers), no_samples=2)
    # scale data
    arr_test = arr_test.transpose()
    arr_train = np.concatenate(cv_train, axis=1).transpose()
    arr_train = arr_train.astype(float)
    arr_train = arr_train[:, 1:]

    arr_train = np.column_stack([scale(arr_train[:, i]) for i in range(arr_train.shape[1])])
    arr_test = np.column_stack([scale(arr_test[:, i]) for i in range(arr_test.shape[1])])
    X_train, y_train = arr_train[:, :-1], arr_train[:, -1]
    X_test, y_test = arr_test[:, :-1], arr_test[:, -1]

    # train model
    mod, hidden_states = get_hmm(X_train, y_train, n_components=3, n_int=1)
    states, statesg = get_hidden_states(hidden_states, y_train)

    fig = plt.figure()
    for i in set(states['states']):
        plt.hist(states[states['states'] == i]['rets'], bins='fd', alpha=.6, label=i)
    plt.legend()
    plt.show()

    sns.violinplot(states, x='states', y='rets')
    plt.plot([-1, 0, 1, 2, 3], [0, 0, 0, 0, 0], color='black')
    plt.show()

    hidden_states = mod.predict(X_test)
    X_test = pd.DataFrame(X_test, columns=cv_cols[1:-1])
    X_test[f'{sel_ind_ticker[0]}_price'] = test[f'{sel_ind_ticker[0]}_price'].iloc[1:].values
    X_test[f'{sel_ind_ticker[0]}'] = test[f'{sel_ind_ticker[0]}'].iloc[1:].values
    X_test['date'] = list(test.index)[1:]
    plot_hmm_states(X_test, hidden_states, f'{sel_ind_ticker[0]}_price', f'{sel_ind_ticker[0]}', 'date')
