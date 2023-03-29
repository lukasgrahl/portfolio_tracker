import pandas as pd
import numpy as np
import datetime
import pprint

import streamlit as st


def apply_datetime_format(x, dt_format: str = None):
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
    perc = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {perc}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
    pass


def get_index(col: str, col_list: list, return_boolean: bool = False):
    assert col in col_list, f'{col} is not in col_list'
    if not return_boolean:
        return col_list.index(col)
    else:
        return np.array([True if i == col_list.index(col) else False for i in range(0, len(col_list))])


def is_outlier(ser: pd.Series, std_factor: float = 5.):
    mu = np.mean(ser)
    sig = np.std(ser) * std_factor
    int_u = mu + sig
    int_l = mu - sig
    return ~((ser <= int_u) & (ser >= int_l))


@st.cache_data()
def train_test_split(df_in: pd.DataFrame, test_size_split: list = [.1]) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits pd.DataFrame alongside axis=0 into train and test sample, assumes most
    recent data to be located on the bottom of the df
    :param test_size_split:
    :param df_in:
    :param test_size: list splits: [.1, .1] for a 10%, 80%, 10% split
    :return: test, train
    """
    assert sum(test_size_split) <= 1, "Test size split exceeds 1"
    df = df_in.copy()

    test_size_split.insert(0, 0)
    if sum(test_size_split) < 1: test_size_split.append(1 - sum(test_size_split))

    test_size_split = np.array(test_size_split) * len(df)
    test_size_split = np.array(np.floor(test_size_split), dtype=int)
    test_size_split = np.cumsum(test_size_split)

    dfs_out = []
    for i in range(1, len(test_size_split)):
        dfs_out.append(df.iloc[test_size_split[i - 1]: test_size_split[i]])

    return tuple(dfs_out)
