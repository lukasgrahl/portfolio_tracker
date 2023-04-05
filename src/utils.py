import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix, roc_auc_score
import streamlit as st
from copy import deepcopy


def apply_datetime_format(x, dt_format: str = None) -> datetime.datetime:
    """
    Function applies a series of non-conflicting time series formats, no need to specify
    :param x: string time stamp
    :param dt_format: datetime time series format
    :return: datetime stamp
    """
    x = str(x)
    if dt_format is None:

        try:
            x = datetime.datetime.strptime(x, "%Y-%m-%d")
            return x
        except ValueError:
            pass

        try:
            x = datetime.datetime.strptime(x, "%m.%d.%Y")
            return x
        except ValueError:
            pass

        try:
            x = datetime.datetime.strptime(x, "%d/%m/%Y")
            return x
        except ValueError:
            pass

        try:
            x = datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S")
            return x
        except ValueError:
            pass

        try:
            x = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            return x
        except ValueError:
            pass

        print("Datetime Assignment failed")
        raise ValueError(x, "Format not in the collection")
    else:
        return datetime.datetime.strptime(x, dt_format)


def get_ARlags(ser: pd.Series, lags: int, suffix: str = 'lag', ret_org_ser: bool = True) -> pd.DataFrame:
    """
    get lagged values for pandas series
    :param ser: input series e.g. prices, returns
    :param lags: number of lags: 1 to lags will be returned
    :param suffix: name suffix for lagged columns
    :param ret_org_ser: boolean on whether to return the non-lagged original series in df
    :return: df with original series and
    """
    df = ser.copy()
    df = pd.DataFrame(df)
    col = ser.name
    for i in range(1, abs(lags) + 1):
        if lags < 0: i = i * -1
        df[f'{col}_{suffix}_{i}'] = ser.shift(i).copy()

    if not ret_org_ser: df.drop(col, axis=1, inplace=True)
    return df.dropna()


def printProgBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Prints progression bar
    :param iteration:
    :param total:
    :param prefix:
    :param suffix:
    :param decimals:
    :param length:
    :param fill:
    :param printEnd:
    :return:
    """
    perc = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {perc}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
    pass


def get_index(x: str, col_list: list, return_boolean: bool = False):
    """
    Gets index of a string element in a list of strings
    :param x: string element
    :param col_list: list of strings
    :param return_boolean: if True function returns a boolean mask array of the list of strings
    :return: either integer index or boolean mask array
    """
    assert x in col_list, f'{x} is not in col_list'
    if not return_boolean:
        return col_list.index(x)
    else:
        return np.array([True if i == col_list.index(x) else False for i in range(0, len(col_list))])


def is_outlier(ser: pd.Series, std_factor: float = 5.):
    """
    Outlier identification function based on standard deviations
    :param ser: data
    :param std_factor: number of standard deviations that form outlier interval
    :return: mask array whether data point is outlier
    """
    mu = np.mean(ser)
    sig = np.std(ser) * std_factor
    int_u = mu + sig
    int_l = mu - sig
    return ~((ser <= int_u) & (ser >= int_l))


def train_test_split(df_in: pd.DataFrame, test_size_split: list = [.1]) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits pd.DataFrame alongside axis=0 into train and test sample, assumes most
    recent data to be located on the bottom of the df
    :param test_size_split:
    :param df_in:
    :param test_size: list splits: [.1, .1] for a 10%, 80%, 10% split
    :return: test, train
    """
    if type(test_size_split) == float:
        test_size_split = [test_size_split]

    df = df_in.copy()
    test_size = deepcopy(test_size_split)
    assert sum(test_size) <= 1, "Test size split exceeds 1"

    test_size.insert(0, 0)
    if sum(test_size) < 1:
        test_size.append(1 - sum(test_size_split))

    test_size = np.array(test_size) * len(df)
    test_size = np.array(np.floor(test_size), dtype=int)
    test_size = np.cumsum(test_size)

    dfs_out = []
    for i in range(1, len(test_size)):
        dfs_out.append(df.iloc[test_size[i - 1]: test_size[i]])

    return tuple(dfs_out)


def get_binary_metric(y_pred: np.array, y_true: np.array, cut_off: float = None):
    """
    Function returns binary classication performance metrics
    :param y_pred: predicted values
    :param y_true: true values
    :param cut_off: Optional: if data is not yet binary, the cut off point will be used to tranform data to boolean
    :return: confustion matrix, roc area under the curve score
    """
    assert y_pred.shape == y_true.shape, "y predicted and y true do not allign in dimension"

    if cut_off is not None:
        y_pred = y_pred >= cut_off
        y_true = y_true >= cut_off

    conf_mat = pd.DataFrame(confusion_matrix(y_true=y_true,
                                             y_pred=y_pred),
                            index=['negative', 'positive'], columns=['true', 'false'])
    conf_mat = conf_mat / len(y_pred)
    roc_score = roc_auc_score(y_true=y_true, y_score=y_pred)
    return conf_mat, roc_score
