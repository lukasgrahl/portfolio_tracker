import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import os
from datetime import timedelta
from itertools import chain

from src.filter import set_up_kalman_filter, kalman_filter, get_ARMA_test
from src.pull_data import train_test_split

from settings import DATA_DIR

if __name__ == "__main__":
    df_rets = pd.read_csv(os.path.join(DATA_DIR, 'returns.csv'), index_col='date').drop('Unnamed: 0', axis=1)
    df_rets = df_rets.iloc[-300:]
    test, train = train_test_split(df_rets, .2)
    train.index = pd.DatetimeIndex(train.index).to_period('D')

    train[f'^GSPC_lead'] = train['^GSPC'].shift(-1)
    train = train.dropna()

    ### ARMA INPUT ####
    endog = ['^GSPC_lead']

    for p, q, exog in [(1, 3, ['AAPL'])]: #, (1, 0, ['AAPL']), (2, 4, ['AAPL', 'GOOG']), (1, 0, [])]:

        print(p, q, exog)

        # get arima output
        p, q, d, ma_resid, arima_params = get_ARMA_test(p, q, train, endog, exog)

        #### Testing Code ####
        # set up filter
        xdim = p + d + q
        zdim = xdim

        T, Q, Z, H, x0, P0, zs, state_vars, zs_index = set_up_kalman_filter(p, q, d, xdim, zdim, train,
                                                                            ma_resid, arima_params, endog, exog)

        X_out, P_out, X_pred, P_pred, LL_out = kalman_filter(xdim, zdim, p, q, d, x0, P0, zs, T, Q, Z, H, state_vars)

        df_xtrue = train[endog].loc[zs_index].copy()
        ind = pd.DatetimeIndex([str(item) for item in zs_index])
        df_xtrue = pd.DataFrame(df_xtrue.values, index=ind, columns=endog)

        ind = pd.DatetimeIndex([*chain([str(item) for item in zs_index], ['2022-12-31'])])
        df_xpred = pd.DataFrame(X_pred[:, 0], index=ind, columns=[f'{endog[0]}_pred'])

        sns.lineplot(df_xtrue)
        plt.show()

        tn, fp, fn, tp = confusion_matrix(y_true=(df_xtrue >= 0), y_pred=(df_xpred.iloc[:-1] >= 0)).ravel()

        # plot results
        # inds, inde = 0, len(train)
        # fig, ax = plt.subplots(figsize=(15, 5))
        # ax.plot(X_pred[inds:inde, 0])
        # ax.plot(train['^GSPC_lead'].values[inds:inde])
        # ax.plot(X_out[inds:inde, 0])
        # ax.legend(['pred', 'true', 'filter'])
        # plt.show()
