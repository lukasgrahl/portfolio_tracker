import numpy as np
from src.utils import assert_dim
from filterpy.kalman import KalmanFilter
import pprint

from src.utils import assert_dim, predict, update
class KalmanFilterBase:
    def __init__(self,
                 xdim,
                 zdim):
        self.xdim = xdim
        self.zdim = zdim

        # exogenous
        self.R = None
        self.Q = None
        # self.z = None
        self.u = None
        self.F = None
        self.x = None
        self.P = None
        self.H = None
        self.B = None

        pass

    # def __repr__(self):
    #     # print(self.__dict__)
    #     return '\n'.join([
    #         'KalmanFilter object',
    #         pprint('xdim', self.xdim),
    #         pprint('zdim', self.zdim),
    #         pprint('F', self.F),
    #         pprint('Q', self.Q),
    #         pprint('R', self.R),
    #         pprint('H', self.H),
    #         pprint('B', self.B),
    #     ])
    #     pass

    def _dim_check(self):
        assert_dim(self.R, self.zdim)
        assert_dim(self.Q, self.xdim)

        assert_dim(self.u, self.xdim, 1)
        assert_dim(self.B, self.xdim)

        assert_dim(self.F, self.xdim)
        assert_dim(self.H, self.zdim, self.xdim)

        # assert_dim(self.x0, self.xdim, 1)
        # assert_dim(self.P0, self.xdim)
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

    def run(self, measurment):
        self._sanity_check()
        assert_dim(self.x, self.xdim, 1)
        assert_dim(self.P, self.xdim, self.xdim)
        # insert sanity check for measurement dimensions

        z = measurment.copy()
        x_b, P_b = predict(x=self.x, P=self.P, F=self.F, B=self.B, u=self.u, Q=self.Q)
        self.x, self.P = update(z=z, x_b=x_b, P_b=P_b, H=self.H, R=self.R)
        return self.x, self.P
