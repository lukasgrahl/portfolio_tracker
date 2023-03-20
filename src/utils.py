import pandas as pd
import numpy as np
import datetime
import pprint
from numba import njit
from scipy import linalg



def assert_dim(A, dim1, dim2=None, isone_d: bool = False):
    if dim2 is None:
        dim2 = dim1

    if type(A) != type(None):
        assert type(A) == np.ndarray, f"{A} is of type {type(A)} should be of type {np.array}"
        if isone_d:
            assert (dim1,) == A.shape, f"{A} should be one dimensional of ({dim1,}) but is {A.shape}"
        else:
            assert (dim1, dim2) == A.shape, f"({dim1}x{dim2}) does not correspond to {A.shape}"
    pass


def predict(x, P, F, B, u, Q):
    x_b = F @ x + B @ u
    P_b = F @ P @ F.transpose() + Q
    return x_b, P_b


def update(z, x_b, P_b, H, R):
    # system uncertainity
    S = H @ P_b @ H.transpose() + R
    # Kalman gain
    K = P_b @ H.transpose() @ np.linalg.inv(S)

    # residual
    y = z - H @ x_b

    # P_t+1
    P_1 = (np.identity(P_b.shape[0]) - K @ H) @ P_b
    # x_t+1
    x_1 = x_b + K @ y

    return x_1, P_1


def get_confidence_interval(mu, var, interval_span: float = 1.96):
    return np.array([mu + interval_span * np.sqrt(var),
                     mu - interval_span * np.sqrt(var)])


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

        pprint("Datetime Assignment failed")
        raise ValueError(x, "Format not in the collection")
    else:
        return datetime.datetime.strptime(x, dt_format)



@njit
def kalman_predict(x, P, T, QN) -> (np.array, np.array):
    # predict
    x_pred = T @ x
    P_pred = T @ P @ T.T + QN
    P_pred = (P_pred + P_pred.T) / 2
    return x_pred, P_pred


def get_ARlags(ser: pd.Series, lags: int, suffix: str = 'lag') -> pd.DataFrame:
    df = ser.copy()
    df = pd.DataFrame(df)
    col = ser.name
    for i in range(1, abs(lags)+1):
        if lags < 0: i = i * -1
        df[f'{col}_{suffix}_{i}'] = ser.shift(i).copy()
    return df.dropna()

def kalman_update(z, x, P, Z, H) -> (np.array, np.array, np.array, bool):
    # residual
    y = z - Z @ x

    # System Uncertainty
    PZT = P @ Z.T
    F = Z @ PZT + H

    # Kalman Gain
    try:
        F_chol = linalg.cholesky(F)
        true_inv = True
    except linalg.LinAlgError:
        F_chol = np.linalg.pinv(F)
        true_inv = False

    K = P @ Z.T @ F_chol

    # update x
    x_up = x + K @ y

    # update P
    I_KZ = np.eye(K.shape[0]) - K @ Z
    P_up = I_KZ @ P @ I_KZ.T + K @ H @ K.T
    P_up = .5 * (P_up + P_up.T)

    # get log-likelihood
    MVN_CONST = np.log(2.0 * np.pi)
    inner_term = linalg.solve_triangular(F_chol, linalg.solve_triangular(F_chol, y, lower=True), lower=True, trans=1)
    n = y.shape[0]
    ll = -0.5 * (n * MVN_CONST + (y.T @ inner_term).ravel()) - np.log(np.diag(F_chol)).sum()
    # ll = np.array([-100])

    return np.array(x_up), np.array(P_up), np.array(ll), true_inv


def kalman_filter(x0, P0, zs, T, Q, Z, H, ma_index: list, ar_index: list):
    assert len(ma_index) == zs.shape[
        -1], f'Moving Average index boolean list does not correspond with measurment dimensions (zs)'
    assert sum(ma_index) <= sum(ar_index), f'Longer moving average component than AR component, not supported'
    for ind_list in [ma_index, ar_index]:
        assert sum([type(item) == bool for item in ind_list]) == len(
            ind_list), f'Moving average index has non-boolean items'

    X_out, P_out, LL_out = [], [], []

    x = x0.copy()
    P = P0.copy()
    true_inv_count = 0

    for i in range(0, len(zs) - 1):
        # kalman predict step
        x_pred, P_pred = kalman_predict(x, P, T, Q)

        # update moving average component for zs[i]
        zs[i, ma_index] = (x_pred[ar_index] - x[ar_index])[0]
        z = zs[i].copy()

        # kalman update step
        x, P, ll, true_inv = kalman_update(z, x_pred, P_pred, Z, H)
        true_inv_count += true_inv

        # if not solved: return np.array(X_out), np.array(P_out), np.array(LL_out)

        X_out.append(x)
        P_out.append(P)
        LL_out.append(ll)

    print('true inverse', true_inv_count / (len(zs) - 1))
    return np.array(X_out), np.array(P_out), np.array(LL_out)



