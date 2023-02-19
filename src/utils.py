import numpy as np


def get_confidence_interval(mu, var, interval_span: float = 1.96):
    return np.array([mu + interval_span * np.sqrt(var),
                     mu - interval_span * np.sqrt(var)])


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
    P_1 = (np.identity(2) - K @ H) @ P_b
    # x_t+1
    x_1 = x_b + K @ y

    return x_1, P_1


def get_confidence_interval(mu, var, interval_span: float = 1.96):
    return np.array([mu + interval_span * np.sqrt(var),
                     mu - interval_span * np.sqrt(var)])