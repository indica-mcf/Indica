from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
import xarray as xr
from xarray import DataArray

from indica.utilities import format_coord
from indica.utilities import format_dataarray


class Profiles:
    def __init__(
        self,
        datatype: str = "electron_temperature",
        xend: float = 1.05,
        xspl: np.ndarray = None,
        coord="poloidal",
        parameters: dict = None,
    ):
        """
        Class to build general profiles

        Parameters
        ----------
        datatype
            Tuple defining what type of profile is to be built
        xspl
            normalised radial grid [0, 1]  on which profile is to be built

        """
        self.y0: float
        self.y1: float
        self.yend: float
        self.peaking: float
        self.wcenter: float
        self.wped: float
        self.parameters: dict = {}

        self.xend = xend
        self.coord = f"rho_{coord}"
        self.x = np.linspace(0, 1, 15) ** 0.7
        self.datatype = datatype
        if xspl is None:
            xspl = format_coord(np.linspace(0, 1.0, 30), self.coord)
        self.xspl = xspl
        self.profile_parameters: list = [
            "y0",
            "y1",
            "yend",
            "wcenter",
            "wped",
            "peaking",
        ]

        if parameters is None:
            parameters = get_defaults(datatype)
        elif {
            "y0",
            "y1",
            "yend",
            "wcenter",
            "wped",
            "peaking",
        } >= set(parameters):
            _parameters = get_defaults(datatype)
            parameters = dict(_parameters, **parameters)

        for k, p in parameters.items():
            setattr(self, k, p)

        self.__call__()

    def set_parameters(self, **kwargs):
        """
        Set any of the shaping parameters
        """
        for k, v in kwargs.items():
            if k in self.parameters:
                setattr(self, k, v)

    def get_parameters(self):
        """
        Set any of the shaping parameters
        """
        parameters_dict: dict = {}
        for k in self.profile_parameters:
            parameters_dict[k] = getattr(self, k)

        return parameters_dict

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

        def gaussian(x, A, B, x_0, w):
            return (A - B) * np.exp(-((x - x_0) ** 2) / (2 * w**2)) + B

        centre = deepcopy(self.y0)
        edge = deepcopy(self.y1)
        wcenter = deepcopy(self.wcenter)
        wped = deepcopy(self.wped)
        peaking = deepcopy(self.peaking)

        # Add additional peaking with respect to reference shape
        peaking2 = 1.0
        if y0_ref is not None:
            if y0_ref < centre:
                peaking2 = centre / y0_ref
        self.peaking2 = peaking2

        if peaking2 > 1:
            centre = y0_ref
            wcenter = wcenter - (peaking2**wcenter_exp - 1)

        centre = centre / peaking

        x = self.x[np.where(self.x <= 1.0)[0]]

        # baseline profile shape
        y = (centre - edge) * (1 - x**wped) + edge

        if debug:
            plt.figure()
            plt.plot(x, y, label="first")

        # add central peaking
        if peaking != 1:
            sigma = wcenter / (np.sqrt(2 * np.log(2)))
            y += gaussian(x, centre * (peaking - 1), 0, 0, sigma)

        if debug:
            plt.plot(x, y, label="peaking")

        # add additional peaking
        if peaking2 != 1:
            sigma = wcenter / (np.sqrt(2 * np.log(2)))
            y += gaussian(x, y[0] * (peaking2 - 1), 0, 0, sigma)

        if debug:
            plt.plot(x, y, label="peaking2")

        if y0_fix:
            y -= edge
            y /= y[0]
            y *= centre - edge
            y += edge

        if debug:
            plt.plot(x, y, label="y0_fix", marker="o")

        x = np.append(x, self.xend)
        y = np.append(y, self.yend)

        if debug:
            plt.plot(x, y, label="0 point at 1.05")

        self.cubicspline = CubicSpline(
            x,
            y,
            0,
            "clamped",
            False,
        )
        _yspl = self.cubicspline(self.xspl)

        coords = {self.coord: self.xspl}
        yspl = format_dataarray(_yspl, self.datatype, coords=coords)
        self.yspl = yspl

        if debug:
            plt.plot(self.xspl, yspl, label="spline")
            plt.legend()

        return yspl

    def plot(self, fig=True):
        self.__call__()
        if fig:
            plt.figure()
        self.yspl.plot()


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
            "ref": True,
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
        _datatype = "temperature_electron"
        print(
            f"\n Profile {datatype} not available "
            f"\n Using '{_datatype}' as default \n"
        )
        datatype = _datatype

    return parameters[datatype]


def profile_scans(plot=False, rho=np.linspace(0, 1.0, 41)):
    Te = Profiles(datatype="electron_temperature", xspl=rho)
    Ne = Profiles(datatype="electron_density", xspl=rho)
    Nimp = Profiles(datatype="impurity_density", xspl=rho)
    Vrot = Profiles(datatype="toroidal_rotation", xspl=rho)

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

    # import pandas as pd
    # to_write = {
    #     "Rho-poloidal": rho,
    #     "Te broad (eV)": Te_broad.yspl.values,
    #     "Te peaked (eV)": Te_peak.yspl.values,
    #     "Ti broad (eV)": Ti_broad.yspl.values,
    #     "Ti peaked (eV)": Ti_peak.yspl.values,
    #     "Ne broad (m^-3)": Ne_broad.yspl.values,
    #     "Ne peaked (m^-3)": Ne_peak.yspl.values,
    # }
    # df = pd.DataFrame(to_write)
    # df.to_csv("/home/marco.sertoli/data/Indica/profiles.csv")


def sawtooth_crash(pre: Profiles, rho_inv: float, volume: DataArray = None):
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

    pre = Profiles(datatype=(identifier, "electron"), xspl=rho)
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
