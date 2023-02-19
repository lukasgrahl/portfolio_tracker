import numpy as np
from src.utils import assert_dim


class KalmanFilterBase:
    def __init__(self,
                 xdim,
                 zdim):
        self.xdim = xdim
        self.zdim = zdim

        # exogenous
        self.R = None
        self.Q = None
        self.zs = None
        self.u = None
        self.F = None
        self.x = None
        self.P = None
        self.H = None
        self.B = None

        pass

    def _dim_check(self):
        assert_dim(self.R, self.zdim)
        assert_dim(self.Q, self.xdim)

        assert_dim(self.u, self.xdim, isone_d=True)
        assert_dim(self.B, self.xdim)

        assert_dim(self.F, self.xdim)
        assert_dim(self.x, self.xdim, isone_d=True)

        assert_dim(self.P, self.xdim)
        assert_dim(self.H, self.zdim, self.xdim)

        if self.zs is not None:
            assert_dim(self.zs, len(self.zs), isone_d=True)
        pass

    def _none_check(self):
        for item in self.__dict__.keys():
            assert type(self.__dict__[item]) != type(None), f"{item} is NoneType"

    def _sanity_check(self):
        self._dim_check()
        self._none_check()
        pass

    def get_dimensions(self):
        return self.xdim, self.zdim

    def predict(self, x, P):
        x_b = self.F @ x + self.B @ self.u
        P_b = self.F @ P @ self.F.transpose() + self.Q
        return x_b, P_b

    def update(self, z, x_b, P_b):
        # system uncertainity
        S = self.H @ P_b @ self.H.transpose() + self.R
        # Kalman gain
        K = P_b @ self.H.transpose() @ np.linalg.inv(S)

        # residual
        y = z - self.H @ x_b

        # P_t+1
        P_1 = (np.identity(self.xdim) - K @ self.H) @ P_b
        # x_t+1
        x_1 = x_b + K @ y
        return x_1, P_1

    def run(self):
        self._sanity_check()

        xb, covb = [], []
        xs, covs = [], []

        x = self.x
        P = self.P
        for i in range(len(self.zs)):
            x_b, P_b = self.predict(x, P)
            xb.append(x_b)
            covb.append(P_b)

            x_1, P_1 = self.update(self.zs[i], x_b, P_b)
            xs.append(x_1)
            covs.append(P_1)

            x = x_1
            P = P_1

        xb = np.array(xb).transpose()
        covb = np.array(covb)
        xs = np.array(xs).transpose()
        covs = np.array(covs)

        return xs, covs
