from copy import deepcopy

from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
from indica.operators.sawtooth_crash import sawtooth_crash
import xarray as xr
from xarray import DataArray

from indica.numpy_typing import LabeledArray
from indica.profilers.profiler_gauss import ProfilerGauss
from indica.utilities import FIG_PATH
from indica.utilities import save_figure
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_rcparams

CMAP = cm.gnuplot2

PARAMETER_LIMITS: dict = {
    "electron_temperature": {
        "y0": (500, 10000),
        "y1": (30, 100),
        "peaking": (1, 5),
        "wcenter": (0.2, 0.4),
        "wped": (2, 10),
    },
    "ion_temperature": {
        "y0": (500, 10000),
        "y1": (30, 100),
        "peaking": (1, 5),
        "wcenter": (0.2, 0.4),
        "wped": (2, 10),
    },
    "electron_density": {
        "y0": (1.0e19, 1.0e20),
        "y1": (1.0e18, 5.0e18),
        "peaking": (1, 3),
        "wcenter": (0.3, 0.4),
        "wped": (2, 10),
    },
    "thermal_neutral_density": {
        "y0": (1.0e13, 1.0e14),
        "y1": (1.0e15, 1.0e16),
        "yend": "y1",
        "peaking": (1, 1),
        "wcenter": (0, 0),
        "wped": (5, 15),
    },
}


def profile_scans_pca(
    parameter_limits: dict = None,
    scans: int = 100,
    datatype: str = "electron_temperature",
    rho: LabeledArray = np.linspace(0, 1.0, 41),
    plot: bool = True,
    save_fig: bool = False,
    fig_path: str = None,
):
    if fig_path is None:
        fig_path = FIG_PATH

    set_plot_rcparams("profiles")

    if parameter_limits is None:
        parameter_limits = PARAMETER_LIMITS[datatype]

    profiler = ProfilerGauss(datatype=datatype, xspl=rho)

    _profile_scans: list = []
    parameter_scans: dict = {}
    parameter_keys = parameter_limits.keys()
    for k in parameter_keys:
        parameter_scans[k] = []

    for iscan in range(scans):
        for k, lim in parameter_limits.items():
            if type(lim) != str:
                param_value = np.random.uniform(lim[0], lim[1])
            else:
                param_value = parameter_scans[lim][-1]

            parameter_scans[k].append(param_value)
            setattr(profiler, k, param_value)
        _profile_scans.append(profiler())

    profile_scans: DataArray = xr.concat(_profile_scans, "scan").assign_coords(
        {"scan": np.arange(scans)}
    )

    for k in parameter_keys:
        profile_scans = profile_scans.assign_coords({k: ("scan", parameter_scans[k])})

    if plot:
        cols = CMAP(np.linspace(0.1, 0.75, scans, dtype=float))
        plt.figure()
        sort_ind = np.argsort(profile_scans.sel(rhop=0)).values
        for i, scan in enumerate(sort_ind):
            profile_scans.sel(scan=scan).plot(
                alpha=0.6,
                color=cols[i],
            )
        plt.title("")
        if datatype == "thermal_neutral_density":
            plt.yscale("log")
        set_axis_sci()
        save_figure(fig_path, f"{datatype}_uniform_scan", save_fig=save_fig)

        for k in parameter_keys:
            counts, bins = np.histogram(getattr(profile_scans, k))
            plt.figure()
            plt.stairs(counts, bins)
            plt.title(f"{k} distribution")

    return profile_scans, profiler


def profile_scans_hda(plot=False, rhop=np.linspace(0, 1.0, 41)):
    Te = ProfilerGauss(datatype="electron_temperature", xspl=rhop)
    Ne = ProfilerGauss(datatype="electron_density", xspl=rhop)
    Nimp = ProfilerGauss(datatype="impurity_density", xspl=rhop)
    Vrot = ProfilerGauss(datatype="toroidal_rotation", xspl=rhop)

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
        Te.ydata.plot(color="black", label="Te")

    # Broad Ti profile without/with Te as reference
    Ti = deepcopy(Te)
    Ti.datatype = "ion_temperature"
    Ti.y0 = 7.0e3
    Ti()
    Ti_list["broad"] = deepcopy(Ti)
    if plot:
        Ti.ydata.plot(linestyle="dashed", color="black", label="Ti no ref")
    Ti(y0_ref=Te.ydata.sel(rhop=0).values)
    if plot:
        Ti.ydata.plot(linestyle="dotted", color="black", label="Ti with ref")

    # Peaked Te profile
    Te.wcenter, Te.wped, Te.y1, Te.peaking = (0.35, 2, 10, 4)
    Te()
    Te_list["peaked"] = deepcopy(Te)
    if plot:
        Te.ydata.plot(color="red", label="Te")

    # Peaked Ti profile without/with Te as reference
    Ti = deepcopy(Te)
    Ti.datatype = "ion_temperature"
    Ti.y0 = 5.0e3
    Ti()
    Ti_list["peaked"] = deepcopy(Ti)
    if plot:
        Ti.ydata.plot(linestyle="dashed", color="red", label="Ti no ref")
    Ti(y0_ref=Te.ydata.sel(rhop=0).values)
    if plot:
        Ti.ydata.plot(linestyle="dotted", color="red", label="Ti with ref")

    Ne.wped = 6
    Ne.y1 = 0.5e19
    Ne()
    Ne_list["broad"] = deepcopy(Ne)
    if plot:
        plt.figure()
        Ne.ydata.plot(color="black")

    Ne.wped = 3.5
    Ne.peaking = 4
    Ne.y1 = 0.1e19
    Ne()
    Ne_list["peaked"] = deepcopy(Ne)
    if plot:
        Ne.ydata.plot(color="red")

    Nimp.wped = 6
    Nimp.y0 = 5.0e16
    Nimp.y1 = 3.0e16
    Nimp.yend = 2.0e16
    Nimp()
    Nimp_list["flat"] = deepcopy(Nimp)
    if plot:
        plt.figure()
        Nimp.ydata.plot(color="black")

    Nimp.peaking = 8
    Nimp.wcenter = 0.2
    Nimp.y1 = 0.5e16
    Nimp.yend = 0.5e16
    Nimp()
    Nimp_list["peaked"] = deepcopy(Nimp)
    if plot:
        Nimp.ydata.plot(color="red")

    Vrot.y1 = 1.0e3
    Vrot.yend = 0.0
    Vrot()
    Vrot_list["broad"] = deepcopy(Vrot)
    if plot:
        plt.figure()
        Vrot.ydata.plot(color="black")

    Vrot.wped = 1
    Vrot.peaking = 2.0
    Vrot()
    Vrot_list["peaked"] = deepcopy(Vrot)
    if plot:
        Vrot.ydata.plot(color="red")

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
    #     "Te broad (eV)": Te_broad.ydata.values,
    #     "Te peaked (eV)": Te_peak.ydata.values,
    #     "Ti broad (eV)": Ti_broad.ydata.values,
    #     "Ti peaked (eV)": Ti_peak.ydata.values,
    #     "Ne broad (m^-3)": Ne_broad.ydata.values,
    #     "Ne peaked (m^-3)": Ne_peak.ydata.values,
    # }
    # df = pd.DataFrame(to_write)
    # df.to_csv("/home/marco.sertoli/data/Indica/profiles.csv")
