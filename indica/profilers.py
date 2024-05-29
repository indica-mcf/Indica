from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
import xarray as xr
from xarray import DataArray

from indica.utilities import format_coord
from indica.utilities import format_dataarray
from abc import ABC


def gaussian(x, A, B, x_0, w):
    return (A - B) * np.exp(-((x - x_0) ** 2) / (2 * w ** 2)) + B


class Profiler(ABC):
    # protocol for profilers to follow

    def __init__(self,
                 parameters: dict = None):

        if parameters is None:
            parameters = {}
        self.parameters = parameters


    def set_parameters(self, **kwargs):
        """
        Set any of the shaping parameters
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_parameters(self):
        """
        Set any of the shaping parameters
        """
        parameters_dict: dict = {}
        for k in self.parameters.keys():
            parameters_dict[k] = getattr(self, k)

        return parameters_dict

    def plot(self, fig=True):
        self.__call__()
        if fig:
            plt.figure()
        self.yspl.plot()

    def __call__(self, *args, **kwargs):
        self.yspl = None



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
            xspl = np.linspace(0, 1.0, 30)
            xspl = DataArray(xspl, coords=[(self.coord, xspl)])
        self.xspl = xspl

        _parameters = get_defaults(datatype)
        if parameters is None:
            parameters = _parameters
        elif {
            "y0",
            "y1",
            "yend",
            "wcenter",
            "wped",
            "peaking",
        } >= set(parameters):
            parameters = dict(_parameters, **parameters)

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
            centre = y0_ref
            wcenter = self.wcenter - (peaking2**wcenter_exp - 1)
        else:
            wcenter = self.wcenter

        center = center / self.peaking

        x = self.x[np.where(self.x <= 1.0)[0]]

        # baseline profile shape
        y_baseline = (center - self.y1) * (1 - x**self.wped) + self.y1

        if self.peaking != 1:  # add central peaking
            sigma = wcenter / (np.sqrt(2 * np.log(2)))
            y_peaking1 = gaussian(x, center * (self.peaking - 1), 0, 0, sigma) + y_baseline
        else:
            y_peaking1 = y_baseline

        if peaking2 != 1:  # add additional peaking
            sigma = wcenter / (np.sqrt(2 * np.log(2)))
            y_peaking2 = gaussian(x, y_peaking1[0] * (peaking2 - 1), 0, 0, sigma) + y_peaking1
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
        coords = [(self.coord, format_coord(self.xspl, self.coord))]
        yspl = format_dataarray(_yspl, self.datatype, coords=coords)
        self.yspl = yspl

        if debug:
            plt.figure()
            plt.plot(x, y_baseline, label="first")
            plt.plot(x, y_peaking1, label="peaking1")
            plt.plot(x, y_peaking2, label="peaking2")
            plt.plot(x, y, label="y0_fix", marker="o")
            plt.plot(self.xspl, yspl, label="spline")
            plt.legend()

        return yspl



def get_defaults(datatype: str) -> dict:
    parameters = {
        "electron_density": {  # (m**-3)
            "y0": 5.0e19,
            "y1": 5.0e18,
            "yend": 2.0e18,
            "peaking": 2,
            "wcenter": 0.4,
            "wped": 6,
        },
        "impurity_density": {  # (m**-3)
            "y0": 5.0e16,
            "y1": 1.0e15,
            "yend": 1.0e15,
            "peaking": 2,
            "wcenter": 0.4,
            "wped": 6,
        },
        "neutral_density": {  # (m**-3)
            "y0": 1.0e13,
            "y1": 1.0e15,
            "yend": 1.0e15,
            "peaking": 1,
            "wcenter": 0,
            "wped": 18,
        },
        "electron_temperature": {  # (eV)
            "y0": 3.0e3,
            "y1": 50,
            "yend": 5,
            "peaking": 1.5,
            "wcenter": 0.35,
            "wped": 3,
        },
        "ion_temperature": {  # (eV)
            "y0": 5.0e3,
            "y1": 50,
            "yend": 5,
            "peaking": 1.5,
            "wcenter": 0.35,
            "wped": 3,
        },
        "toroidal_rotation": {  # (rad/s)
            "y0": 500.0e3,
            "y1": 10.0e3,
            "yend": 0.0,
            "peaking": 1.5,
            "wcenter": 0.35,
            "wped": 3,
        },
    }

    if datatype not in parameters.keys():
        raise ValueError(
            f"\n Profile {datatype} not available "
        )

    return parameters[datatype]



def profile_scans(plot=False, rho=np.linspace(0, 1.0, 41)):
    Te = ProfilerGauss(datatype="electron_temperature", xspl=rho)
    Ne = ProfilerGauss(datatype="electron_density", xspl=rho)
    Nimp = ProfilerGauss(datatype="impurity_density", xspl=rho)
    Vrot = ProfilerGauss(datatype="toroidal_rotation", xspl=rho)

    Te_list = {}
    Ti_list = {}
    Ne_list = {}
    Nimp_list = {}
    Vrot_list = {}

    # Broad Te profile
    Te.y1 = 30
    Te.wped = 3
    Te.wcenter = 0.35
    Te()
    Te_list["broad"] = deepcopy(Te)
    if plot:
        plt.figure()
        Te.yspl.plot(color="black", label="Te")

    # Broad Ti profile without/with Te as reference
    Ti = deepcopy(Te)
    Ti.datatype = ("temperature", "ion")
    Ti.y0 = 7.0e3
    Ti()
    Ti_list["broad"] = deepcopy(Ti)
    if plot:
        Ti.yspl.plot(linestyle="dashed", color="black", label="Ti no ref")
    Ti(y0_ref=Te.yspl.sel(rho_poloidal=0).values)
    if plot:
        Ti.yspl.plot(linestyle="dotted", color="black", label="Ti with ref")

    # Peaked Te profile
    Te.wcenter, Te.wped, Te.y1, Te.peaking = (0.35, 2, 10, 4)
    Te()
    Te_list["peaked"] = deepcopy(Te)
    if plot:
        Te.yspl.plot(color="red", label="Te")

    # Peaked Ti profile without/with Te as reference
    Ti = deepcopy(Te)
    Ti.datatype = ("temperature", "ion")
    Ti.y0 = 5.0e3
    Ti()
    Ti_list["peaked"] = deepcopy(Ti)
    if plot:
        Ti.yspl.plot(linestyle="dashed", color="red", label="Ti no ref")
    Ti(y0_ref=Te.yspl.sel(rho_poloidal=0).values)
    if plot:
        Ti.yspl.plot(linestyle="dotted", color="red", label="Ti with ref")

    Ne.wped = 6
    Ne.y1 = 0.5e19
    Ne()
    Ne_list["broad"] = deepcopy(Ne)
    if plot:
        plt.figure()
        Ne.yspl.plot(color="black")

    Ne.wped = 3.5
    Ne.peaking = 4
    Ne.y1 = 0.1e19
    Ne()
    Ne_list["peaked"] = deepcopy(Ne)
    if plot:
        Ne.yspl.plot(color="red")

    Nimp.wped = 6
    Nimp.y0 = 5.0e16
    Nimp.y1 = 3.0e16
    Nimp.yend = 2.0e16
    Nimp()
    Nimp_list["flat"] = deepcopy(Nimp)
    if plot:
        plt.figure()
        Nimp.yspl.plot(color="black")

    Nimp.peaking = 8
    Nimp.wcenter = 0.2
    Nimp.y1 = 0.5e16
    Nimp.yend = 0.5e16
    Nimp()
    Nimp_list["peaked"] = deepcopy(Nimp)
    if plot:
        Nimp.yspl.plot(color="red")

    Vrot.y1 = 1.0e3
    Vrot.yend = 0.0
    Vrot()
    Vrot_list["broad"] = deepcopy(Vrot)
    if plot:
        plt.figure()
        Vrot.yspl.plot(color="black")

    Vrot.wped = 1
    Vrot.peaking = 2.0
    Vrot()
    Vrot_list["peaked"] = deepcopy(Vrot)
    if plot:
        Vrot.yspl.plot(color="red")

    return {
        "Te": Te_list,
        "Ti": Ti_list,
        "Ne": Ne_list,
        "Nimp": Nimp_list,
        "Vrot": Vrot_list,
    }


def sawtooth_crash(pre: ProfilerGauss, rho_inv: float, volume: DataArray = None):
    """
    Model a sawtooth crash, without resetting the internal quantities

    If volume in not None, then volume integral is conserved adapting the edge shape

    Parameters
    ----------
    rho_inv
        Inversion radius
    volume
        Plasma volume

    """
    post = deepcopy(pre)

    if volume is None:
        volume = DataArray(0.85 * pre.xspl**3, coords=[("rho_poloidal", pre.xspl)])
    vol = volume.interp(rho_poloidal=pre.yspl.rho_poloidal)
    vol_int_pre = np.trapz(pre.yspl, vol)

    x = pre.x[np.where(pre.x <= 1.0)[0]]

    rho = post.yspl.rho_poloidal
    inv_ind = np.max(np.where(rho <= rho_inv)[0])
    for rind in np.arange(inv_ind, rho.size):
        y = xr.where(
            rho <= rho[rind], post.yspl.sel(rho_poloidal=rho[inv_ind]), post.yspl
        )
        vol_int_post = np.trapz(y, vol)
        if vol_int_post >= vol_int_pre:
            break

    y = xr.where(rho != rho[rind], y, (y[rind] + y[rind + 1]) / 2)
    y = y.interp(rho_poloidal=x)

    x = np.append(x, pre.xend)
    y = np.append(y, pre.yend)
    cubicspline = CubicSpline(
        x,
        y,
        0,
        "clamped",
        False,
    )
    post.yspl.values = cubicspline(pre.xspl)
    vol_int_post = np.trapz(post.yspl, vol)

    print(f"Vol-int: {vol_int_pre}, {vol_int_post}")

    return post


def density_crash(
    los_avrg=2.8e19,
    drop=0.9,
    rho=np.linspace(0, 1, 20),
    rho_inv=0.4,
    identifier="density",
):
    volume = DataArray(0.85 * rho**3, coords=[("rho_poloidal", rho)])

    pre = ProfilerGauss(datatype=(identifier, "electron"), xspl=rho)
    pre.wcenter = rho_inv / 1.5
    pre()

    plt.figure()

    drop_arr = []
    scan = np.linspace(1.0, 5.0, 21)
    for s in scan:
        pre.peaking = s
        pre()
        pre.y0 *= los_avrg / pre.yspl.mean().values
        pre()

        post = deepcopy(pre)
        pre.yspl.plot()
        _post = post.sawtooth_crash(rho_inv, volume)
        drop_arr.append(_post.mean().values / pre.yspl.mean().values)

        _post.plot(linestyle="dashed")

    mn_ind = np.argmin(np.abs(np.array(drop_arr) - drop))
    pre.peaking = scan[mn_ind]
    pre()
    pre.y0 *= los_avrg / pre.yspl.mean().values
    pre()

    post = deepcopy(pre)
    _post = post.sawtooth_crash(rho_inv, volume)

    pre.yspl.plot(marker="o", color="black")
    _post.plot(marker="o", color="red")

    print(drop, drop_arr[mn_ind])

    return pre.yspl, _post
