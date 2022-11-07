import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline
from xarray import DataArray
from copy import deepcopy


class Profiles:
    def __init__(
        self, datatype: tuple = ("temperature", "electron"), xspl: np.ndarray = None, xend:float=1.05):
        """
        Class to build general profiles e.g. temperature, density, rotation and neutral density

        Parameters
        ----------
        datatype
            Tuple defining what type of profile is to be built
        xspl
            normalised radial grid [0, 1]  on which profile is to be built
        xend
            last point in the grid for extrapolation outside of the separatrix
        """

        self.x = np.linspace(0, 1, 15) ** 0.7
        if xspl is None:
            xspl = np.linspace(0, 1.0, 30)
        self.xspl = xspl
        self.xend = xend
        self.yend = 0

        self.datatype = datatype

        params = get_defaults(datatype)
        for k, p in params.items():
            setattr(self, k, p)

        self.build_profile()

    def build_profile(
        self,
        y0_fix=False,
        y0_ref=None,
        wcenter_exp=0.05,
        coord="rho_poloidal",
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
            return (A - B) * np.exp(-((x - x_0) ** 2) / (2 * w ** 2)) + B

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
            wcenter = wcenter - (peaking2 ** wcenter_exp - 1)

        centre = centre / peaking

        x = self.x[np.where(self.x <= 1.0)[0]]

        # baseline profile shape
        y = (centre - edge) * (1 - x ** wped) + edge

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

        self.cubicspline = CubicSpline(x, y, 0, "clamped", False,)
        yspl = self.cubicspline(self.xspl)

        if debug:
            plt.plot(self.xspl, yspl, label="spline")
            plt.legend()

        yspl = DataArray(yspl, coords=[(coord, self.xspl)])
        attrs = {"datatype": self.datatype}
        name = self.datatype[1] + "_" + self.datatype[0]
        yspl.name = name
        yspl.attrs = attrs

        self.yspl = yspl

    def sawtooth_crash(self, rho_inv: float, volume: DataArray = None):
        """
        Model a sawtooth crash
        If volume in not None, then volume integral is conserved adapting the edge shape

        New profile will the result after the crash

        Parameters
        ----------
        rho_inv
            Inversion radius
        volume
            Plasma volume

        """
        vol_int_pre = None
        if volume is not None:
            self.vol = volume.interp(rho_poloidal=self.yspl.rho_poloidal)

        if self.vol is not None:
            vol_int_pre = np.trapz(self.yspl, self.vol)

        x = self.x[np.where(self.x <= 1.0)[0]]

        rho = self.yspl.rho_poloidal
        inv_ind = np.max(np.where(rho <= rho_inv)[0])
        for rind in np.arange(inv_ind, rho.size):
            # rind += 1
            y = xr.where(
                rho <= rho[rind], self.yspl.sel(rho_poloidal=rho[inv_ind]), self.yspl
            )
            vol_int_post = np.trapz(y, self.vol)
            # y.plot()
            # print(vol_int_pre - vol_int_post)
            if vol_int_post >= vol_int_pre:
                break

        y = xr.where(rho != rho[rind], y, (y[rind] + y[rind + 1]) / 2)
        y = y.interp(rho_poloidal=x)

        x = np.append(x, self.xend)
        y = np.append(y, self.yend)
        cubicspline = CubicSpline(x, y, 0, "clamped", False,)
        self.yspl.values = cubicspline(self.xspl)
        vol_int_post = np.trapz(self.yspl, self.vol)

        print(f"Vol-int: {vol_int_pre}, {vol_int_post}")

    def plot(self, fig=True):
        if fig:
            plt.figure()
        self.yspl.plot()


def get_defaults(datatype: tuple) -> dict:
    identifier = f"{datatype[0]}_{datatype[1]}"
    parameters = {
        "density_electron": {
            "y0": 5.0e19,
            "y1": 5.e18,
            "yend": 2.e18,
            "peaking": 2,
            "wcenter": 0.4,
            "wped": 6,
        },
        "density_impurity": {
            "y0": 5.0e16,
            "y1": 1.0e16,
            "yend": 1.e15,
            "peaking": 2,
            "wcenter": 0.4,
            "wped": 6,
        },
        "density_thermal_neutrals": {
            "y0": 1.0e13,
            "y1": 1.0e15,
            "yend": 1.0e15,
            "peaking": 1,
            "wcenter": 0,
            "wped": 18,
        },
        "temperature_electron": {
            "y0": 3.0e3,
            "y1": 50,
            "yend": 5.,
            "peaking": 1.5,
            "wcenter": 0.35,
            "wped": 3,
        },
        "temperature_ion": {
            "y0": 5.0e3,
            "y1": 50,
            "yend": 5.,
            "peaking": 1.5,
            "wcenter": 0.35,
            "wped": 3,
            "ref": True,
        },
        "rotation_toroidal": {
            "y0": 500.0e3,
            "y1": 10.0e3,
            "yend": 0.0,
            "peaking": 1.5,
            "wcenter": 0.35,
            "wped": 3,
        },
    }

    if identifier not in parameters.keys():
        print(
            f"\n Profile {identifier} not available \n Using 'temperature_electron' as default \n"
        )
        identifier = "temperature_electron"

    return parameters[identifier]


def profile_scans(plot=False, rho=np.linspace(0, 1.0, 41)):
    Te = Profiles(datatype=("temperature", "electron"), xspl=rho)
    Ne = Profiles(datatype=("density", "electron"), xspl=rho)
    Nimp = Profiles(datatype=("density", "impurity"), xspl=rho)
    Vrot = Profiles(datatype=("rotation", "toroidal"), xspl=rho)

    Te_list = {}
    Ti_list = {}
    Ne_list = {}
    Nimp_list = {}
    Vrot_list = {}

    # Broad Te profile
    Te.y1 = 30
    Te.wped = 3
    Te.wcenter = 0.35
    Te.build_profile()
    Te_list["broad"] = deepcopy(Te)
    if plot:
        plt.figure()
        Te.yspl.plot(color="black", label="Te")

    # Broad Ti profile without/with Te as reference
    Ti = deepcopy(Te)
    Ti.datatype = ("temperature", "ion")
    Ti.y0 = 7.0e3
    Ti.build_profile()
    Ti_list["broad"] = deepcopy(Ti)
    if plot:
        Ti.yspl.plot(linestyle="dashed", color="black", label="Ti no ref")
    Ti.build_profile(y0_ref=Te.yspl.sel(rho_poloidal=0).values)
    if plot:
        Ti.yspl.plot(linestyle="dotted", color="black", label="Ti with ref")

    # Peaked Te profile
    Te.wcenter, Te.wped, Te.y1, Te.peaking = (0.35, 2, 10, 4)
    Te.build_profile()
    Te_list["peaked"] = deepcopy(Te)
    if plot:
        Te.yspl.plot(color="red", label="Te")

    # Peaked Ti profile without/with Te as reference
    Ti = deepcopy(Te)
    Ti.datatype = ("temperature", "ion")
    Ti.y0 = 5.0e3
    Ti.build_profile()
    Ti_list["peaked"] = deepcopy(Ti)
    if plot:
        Ti.yspl.plot(linestyle="dashed", color="red", label="Ti no ref")
    Ti.build_profile(y0_ref=Te.yspl.sel(rho_poloidal=0).values)
    if plot:
        Ti.yspl.plot(linestyle="dotted", color="red", label="Ti with ref")

    Ne.wped = 6
    Ne.y1 = 0.5e19
    Ne.build_profile()
    Ne_list["broad"] = deepcopy(Ne)
    if plot:
        plt.figure()
        Ne.yspl.plot(color="black")

    Ne.wped = 3.5
    Ne.peaking = 4
    Ne.y1 = 0.1e19
    Ne.build_profile()
    Ne_list["peaked"] = deepcopy(Ne)
    if plot:
        Ne.yspl.plot(color="red")

    Nimp.wped = 6
    Nimp.y0 = 5.0e16
    Nimp.y1 = 3.0e16
    Nimp.yend = 2.0e16
    Nimp.build_profile()
    Nimp_list["flat"] = deepcopy(Nimp)
    if plot:
        plt.figure()
        Nimp.yspl.plot(color="black")

    Nimp.peaking = 8
    Nimp.wcenter = 0.2
    Nimp.y1 = 0.5e16
    Nimp.yend = 0.5e16
    Nimp.build_profile()
    Nimp_list["peaked"] = deepcopy(Nimp)
    if plot:
        Nimp.yspl.plot(color="red")

    Vrot.y1 = 1.0e3
    Vrot.yend = 0.0
    Vrot_list["broad"] = deepcopy(Vrot)
    if plot:
        plt.figure()
        Vrot.yspl.plot(color="black")

    Vrot.wped = 1
    Vrot.peaking = 2.0
    Vrot.build_profile()
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


def density_crash(
    los_avrg=2.8e19,
    drop=0.9,
    rho=np.linspace(0, 1, 20),
    rho_inv=0.4,
    identifier="density",
):
    volume = DataArray(0.85 * rho ** 3, coords=[("rho_poloidal", rho)])

    pre = Profiles(datatype=(identifier, "electron"), xspl=rho)
    pre.wcenter = rho_inv / 1.5
    pre.build_profile()

    plt.figure()

    drop_arr = []
    scan = np.linspace(1.0, 5.0, 21)
    for s in scan:
        pre.peaking = s
        pre.build_profile()
        pre.y0 *= los_avrg / pre.yspl.mean().values
        pre.build_profile()

        post = deepcopy(pre)
        pre.yspl.plot()
        post.sawtooth_crash(rho_inv, volume)
        drop_arr.append(post.yspl.mean().values / pre.yspl.mean().values)

        post.yspl.plot(linestyle="dashed")

    mn_ind = np.argmin(np.abs(np.array(drop_arr) - drop))
    pre.peaking = scan[mn_ind]
    pre.build_profile()
    pre.y0 *= los_avrg / pre.yspl.mean().values
    pre.build_profile()

    post = deepcopy(pre)
    post.sawtooth_crash(rho_inv, volume)

    pre.yspl.plot(marker="o", color="black")
    post.yspl.plot(marker="o", color="red")

    print(drop, drop_arr[mn_ind])

    return pre, post
