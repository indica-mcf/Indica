from abc import ABC

import flatdict
import matplotlib.pylab as plt
import numpy as np
from hydra import initialize_config_module, compose
from omegaconf import OmegaConf
from scipy.interpolate import CubicSpline
import xarray as xr

from indica.utilities import format_coord
from indica.utilities import format_dataarray


def gaussian(x, A, B, x_0, w):
    return (A - B) * np.exp(-((x - x_0) ** 2) / (2 * w**2)) + B


class Profiler(ABC):
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


def get_defaults_for_profiler_gauss(datatype="electron_temperature", config_name="profiler_gauss"):
    """
    Loads config for default parameter values
    """
    with initialize_config_module(
        version_base = None, config_module="indica.configs.profilers"
    ):
        cfg = compose(config_name=config_name)
    return OmegaConf.to_container(cfg[datatype])


class ProfilerGauss(Profiler):
    def __init__(
        self,
        datatype: str = "electron_temperature",
        xend: float = 1.05,
        xspl: np.ndarray = None,
        coord="poloidal",
        parameters: dict = None,
    ):
        """
        Class to build general gaussian profiles

        Parameters
        ----------
        datatype
            str defining what type of profile is to be built
        xspl
            normalised radial grid [0, 1] on which profile is to be built

        """
        super().__init__(parameters)
        self.y0: float = None
        self.y1: float = None
        self.yend: float = None
        self.peaking: float = None
        self.wcenter: float = None
        self.wped: float = None
        self.xend = xend
        self.coord = f"rho_{coord}"
        self.x = np.linspace(0, 1, 15) ** 0.7
        self.datatype = datatype
        if xspl is None:
            xspl = format_coord(np.linspace(0, 1.0, 30), self.coord)
        self.xspl = xspl

        default_parameters = get_defaults_for_profiler_gauss(datatype=datatype)
        parameters = dict(default_parameters, **self.parameters)  # Overwrites defaults
        self.set_parameters(**parameters)

    def __call__(
        self,
        y0_fix=False,
        y0_ref=None,
        wcenter_exp=0.05,
        debug=False,
    ):
        """
        Builds the profiles using the parameters set

        Parameters
        ----------
        y0_fix
            re-scale new profile to have central value = y0
        y0_ref
            reference y0 value
        wcenter_exp
            exponent for the additional peaking if y0_ref is not None

        Returns
        -------

        """

        # Add additional peaking with respect to reference shape
        peaking2 = 1.0
        if y0_ref is not None:
            if y0_ref < self.y0:
                peaking2 = self.y0 / y0_ref
        self.peaking2 = peaking2

        center = self.y0
        edge = self.y1

        if peaking2 > 1:
            center = y0_ref
            wcenter = self.wcenter - (peaking2**wcenter_exp - 1)
        else:
            wcenter = self.wcenter

        center = center / self.peaking

        x = self.x[np.where(self.x <= 1.0)[0]]

        # baseline profile shape
        y_baseline = (center - self.y1) * (1 - x**self.wped) + self.y1

        if self.peaking != 1:  # add central peaking
            sigma = wcenter / (np.sqrt(2 * np.log(2)))
            y_peaking1 = (
                gaussian(x, center * (self.peaking - 1), 0, 0, sigma) + y_baseline
            )
        else:
            y_peaking1 = y_baseline

        if peaking2 != 1:  # add additional peaking
            sigma = wcenter / (np.sqrt(2 * np.log(2)))
            y_peaking2 = (
                gaussian(x, y_peaking1[0] * (peaking2 - 1), 0, 0, sigma) + y_peaking1
            )
        else:
            y_peaking2 = y_peaking1

        if y0_fix:
            y = y_peaking2 - edge
            y = y / y[0]
            y = y * (center - edge)
            y = y + edge

        else:
            y = y_peaking2

        x = np.append(x, self.xend)
        y = np.append(y, self.yend)

        self.cubicspline = CubicSpline(
            x,
            y,
            0,
            "clamped",
            False,
        )
        _yspl = self.cubicspline(self.xspl)
        coords = {self.coord: self.xspl}
        ydata = format_dataarray(_yspl, self.datatype, coords=coords)
        self.ydata = ydata

        if debug:
            plt.figure()
            plt.plot(x, y_baseline, label="first")
            plt.plot(x, y_peaking1, label="peaking1")
            plt.plot(x, y_peaking2, label="peaking2")
            plt.plot(x, y, label="y0_fix", marker="o")
            plt.plot(self.xspl, ydata, label="spline")
            plt.legend()

        return ydata


class ProfilerBasis(Profiler):
    """
    Class to build pca profiles from basis functions

    Parameters
    ----------
    radial_grid
        normalised radial grid [0, 1]  on which profile is to be built
    """

    def __init__(
        self,
        basis_functions: np.ndarray,
        bias: np.ndarray,
        ncomps: int = 2,
        radial_grid: np.ndarray = None,
        coord="rho_poloidal",
        parameters: dict = None,
    ):
        super().__init__(parameters)
        self.basis_functions = basis_functions
        self.bias = bias
        self.ncomps = ncomps
        self.radial_grid = radial_grid
        self.coord = coord

        # Weights have to be dynamically assigned as attributes, so they can be accessed
        # / changed by the profiler
        for icomp in range(self.ncomps):
            param_name = f"weight_{icomp + 1}"
            self.parameters[param_name] = 0
            setattr(self, param_name, 0)

    def construct_profile(
        self,
    ):
        weights = np.stack(
            [weight for weight_name, weight in self.parameters.items()], axis=-1
        ).T
        return np.dot(weights, self.basis_functions) + self.bias

    def __call__(
        self,
    ):
        """
        Builds the profile from basis functions using the parameters set
        """
        y = self.construct_profile()
        self.ydata = xr.DataArray(y, coords=[(self.coord, self.radial_grid.data)])
        return self.ydata


def initialise_gauss_profilers(
    xspl: np.ndarray, profile_params: dict = None, profiler_names: list = None
):
    # Should profilers be a dataclass or named tuple rather than bare dictionary
    if profile_params is None:
        profile_params = DEFAULT_PROFILE_PARAMS
    flat_profile_params = flatdict.FlatDict(profile_params, ".")

    if profiler_names is None:
        profile_names = flat_profile_params.as_dict().keys()
    else:
        profile_names = profiler_names

    profilers = {
        profile_name: ProfilerGauss(
            datatype=profile_name.split(":")[0],
            parameters=flat_profile_params[profile_name],
            xspl=xspl,
        )
        for profile_name in profile_names
    }
    return profilers

