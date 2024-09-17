from abc import ABC
import matplotlib.pylab as plt
import numpy as np


def gaussian(x, A, B, x_0, w):
    return (A - B) * np.exp(-((x - x_0) ** 2) / (2 * w**2)) + B


class ProfilerBase(ABC):
    # protocol for profilers to follow

    def __init__(self, parameters: dict = None):
        if parameters is None:
            parameters = {}
        self.parameters = parameters

    def set_parameters(self, **kwargs):
        """
        Set any of the shaping parameters
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.parameters.update(**kwargs)

    def get_parameters(self):
        """
        get all the shaping parameters
        """
        return {key: getattr(self, key) for key in self.parameters.keys()}

    def plot(self, fig=True):
        self.__call__()
        if fig:
            plt.figure()
        self.ydata.plot()

    def __call__(self, *args, **kwargs):
        self.ydata = None
