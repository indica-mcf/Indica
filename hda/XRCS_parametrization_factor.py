"""Read and plot time evolution of various quantities
at identical times in a discharge over a defined pulse range

Example call:

    import hda.regression_analysis as regr
    regr_data = regr.Database(reload=True, pulse_start=8207, pulse_end=9486)
    regr_data()
    regr.plot(regr_data)

    latest_pulse = ...
    regr.add_pulses(regr_data, latest_pulse)


TODO: add the following quantities
Ti/Te, Vloop, all NBI, li, betaP, geometry (volm, elongation, ..)
"""

from copy import deepcopy
import getpass
import pickle

import hda.fac_profiles as fac
from hda.forward_models import Spectrometer
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray
from xarray import Dataset
import os

from indica.readers import ST40Reader
from indica.readers import ADASReader

# First pulse after Boronisation / GDC
BORONISATION = [8441, 8537]
GDC = [8545]
# for p in np.arange(8547, 8560+1):
#     GDC.append(p)
GDC = np.array(GDC) - 0.5

plt.ion()

def build_parametrization(savefig=False, write=False):
    temp_ratio = simulate_xrcs(write=write)

    plt.figure()
    for i in range(len(temp_ratio)):
        plt.plot(temp_ratio[i].te0, temp_ratio[i].te_xrcs)

    plt.plot(temp_ratio[0].te0, temp_ratio[0].te0, "--k", label="Central Te")
    plt.legend()
    add_to_plot(
    "T$_e$(0)",
    "T$_{e,i}$(XRCS)",
    "XRCS measurement vs. Central Te",
    )
    if savefig:
        save_figure(fig_path, f"{fig_name}_XRCS_Te0_parametrization")

    plt.figure()
    for i in range(len(temp_ratio)):
        el_temp = temp_ratio[i].attrs["el_temp"]
        plt.plot(
        el_temp.rho_poloidal,
        el_temp.sel(t=el_temp.t.mean(), method="nearest") / 1.0e3,
        )

    plt.legend()
    add_to_plot(
        "rho_poloidal",
        "T$_e$ (keV)",
        "Temperature profiles",
    )
    if savefig:
        save_figure(fig_path, f"{fig_name}_XRCS_parametrization_temperatures")

    plt.figure()
    for i in range(len(temp_ratio)):
        el_dens = temp_ratio[i].attrs["el_dens"]
        plt.plot(
            el_dens.rho_poloidal,
            el_dens.sel(t=el_dens.t.mean(), method="nearest") / 1.0e3,
        )

    plt.legend()
    add_to_plot(
        "rho_poloidal",
        "n$_e$ (10$^{19}$)",
        "Density profiles",
    )
    if savefig:
        save_figure(fig_path, f"{fig_name}_XRCS_parametrization_densities")


def save_figure(fig_path, fig_name, orientation="landscape", ext=".jpg"):
    plt.savefig(
        fig_path + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )


def simulate_xrcs(pickle_file = "XRCS_temperature_parametrization.pkl", write=False):
    print("Simulating XRCS measurement for Te(0) re-scaling")

    adasreader = ADASReader()
    xrcs = Spectrometer(
        adasreader,
        "ar",
        "16",
        transition="(1)1(1.0)-(1)0(0.0)",
        wavelength=4.0,
    )

    time = np.linspace(0, 1, 50)
    te_0 = np.linspace(0.5e3, 8.0e3, 50)  # central temperature
    te_sep = 50  # separatrix temperature

    # Test two different profile shapes: flat (Ohmic) and slightly peaked (NBI)
    peaked = profiles_peaked()
    broad = profiles_broad()

    temp = [broad.te, peaked.te]
    dens = [broad.ne, peaked.ne]

    el_temp = deepcopy(temp)
    el_dens = deepcopy(dens)

    for i in range(len(dens)):
        el_dens[i] = el_dens[i].expand_dims({"t": len(time)})
        el_dens[i] = el_dens[i].assign_coords({"t": time})
        el_temp[i] = el_temp[i].expand_dims({"t": len(time)})
        el_temp[i] = el_temp[i].assign_coords({"t": time})
        temp_tmp = deepcopy(el_temp[i])
        for it, t in enumerate(time):
            temp_tmp.loc[dict(t=t)] = scale_prof(temp[i], te_0[it], te_sep).values
        el_temp[i] = temp_tmp

    temp_ratio = []
    for idens in range(len(dens)):
        for itemp in range(len(dens)):
            xrcs.simulate_measurements(el_dens[idens], el_temp[itemp], el_temp[itemp])

            tmp = DataArray(
                te_0 / xrcs.el_temp.values, coords=[("te_xrcs", xrcs.el_temp.values)]
            )
            tmp.attrs = {"el_temp": el_temp[itemp], "el_dens": el_dens[idens]}
            temp_ratio.append(tmp.assign_coords(te0=("te_xrcs", te_0)))

    if write:
        pickle.dump(temp_ratio, open(f"/home/marco.sertoli/data/{pickle_file}", "wb"))

    return temp_ratio


def scale_prof(profile, centre, separatrix):
    scaled = profile - profile.sel(rho_poloidal=1.0)
    scaled /= scaled.sel(rho_poloidal=0.0)
    scaled = scaled * (centre - separatrix) + separatrix

    return scaled


def profiles_broad(te_sep=50):
    rho = np.linspace(0, 1, 100)
    profs = fac.Plasma_profs(rho)

    ne_0 = 5.0e19
    profs.ne = profs.build_density(
        y_0=ne_0,
        y_ped=ne_0,
        x_ped=0.88,
        w_core=4.0,
        w_edge=0.1,
        datatype=("density", "electron"),
    )
    te_0 = 1.0e3
    profs.te = profs.build_temperature(
        y_0=te_0,
        y_ped=50,
        x_ped=1.0,
        w_core=0.6,
        w_edge=0.05,
        datatype=("temperature", "electron"),
    )
    profs.te = scale_prof(profs.te, te_0, te_sep)

    ti_0 = 1.0e3
    profs.ti = profs.build_temperature(
        y_0=ti_0,
        y_ped=50,
        x_ped=1.0,
        w_core=0.6,
        w_edge=0.05,
        datatype=("temperature", "ion"),
    )
    profs.ti = scale_prof(profs.ti, ti_0, te_sep)

    return profs


def profiles_peaked(te_sep=50):
    rho = np.linspace(0, 1, 100)
    profs = fac.Plasma_profs(rho)

    # slight central peaking and lower separatrix
    ne_0 = 5.0e19
    profs.ne = profs.build_density(
        y_0=ne_0,
        y_ped=ne_0 / 1.25,
        x_ped=0.85,
        w_core=4.0,
        w_edge=0.1,
        datatype=("density", "electron"),
    )
    te_0 = 1.0e3
    profs.te = profs.build_temperature(
        y_0=te_0,
        y_ped=50,
        x_ped=1.0,
        w_core=0.4,
        w_edge=0.05,
        datatype=("temperature", "electron"),
    )
    profs.te = scale_prof(profs.te, te_0, te_sep)

    ti_0 = 1.0e3
    profs.ti = profs.build_temperature(
        y_0=ti_0,
        y_ped=50,
        x_ped=1.0,
        w_core=0.4,
        w_edge=0.05,
        datatype=("temperature", "ion"),
    )
    profs.ti = scale_prof(profs.ti, ti_0, te_sep)

    return profs

def add_to_plot(xlab, ylab, tit, legend=True, vlines=False):
    if vlines:
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    if legend:
        plt.legend()
