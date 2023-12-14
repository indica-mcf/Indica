"""Definition of weighting functions for line of sight spots.
"""

import getpass
from typing import cast
from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray
from xarray import Dataset
from xarray import zeros_like

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .abstractconverter import find_wall_intersections
from .line_of_sight import LineOfSightTransform
from ..numpy_typing import LabeledArray
from ..numpy_typing import OnlyArray


class SpotWeightings:
    """

    """

    def __init__(
        self,
        los_transform: LineOfSightTransform,
        dist_func: str,
        sigma_w: float = 0.01,
        sigma_v: float = 0.01,
        p_w: float = 1.0,
        p_v: float = 1.0,
    ):
        self.los_transform = los_transform
        self.dist_func = dist_func
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v
        self.p_w = p_w
        self.p_v = p_v

        n_w = int(np.sqrt(los_transform.beamlets))
        n_v = int(np.sqrt(los_transform.beamlets))
        grid_w = np.linspace(
            -los_transform.spot_width / 2,
            los_transform.spot_width / 2,
            n_w * 2 + 1,
            dtype=float,
        )
        grid_w = grid_w[1::2]
        grid_v = np.linspace(
            -los_transform.spot_height / 2,
            los_transform.spot_height / 2,
            n_v * 2 + 1,
            dtype=float,
        )
        grid_v = grid_v[1::2]
        W, V = np.meshgrid(grid_w, grid_v)

        self.delta_w = grid_w
        self.delta_v = grid_v
        self.W = W
        self.V = V

        if dist_func.lower() == "gaussian":
            print("GAUSSIAN DRAGON")
            self.super_gaussian()

        else:
            raise ValueError("dist_func does not exist")

    def super_gaussian(self):
        self.weightings = np.exp(
            -(((self.W) ** 2 / (2 * self.sigma_w ** 2)) ** self.p_w)
            - ((self.V) ** 2 / (2 * self.sigma_v ** 2)) ** self.p_v
        )
