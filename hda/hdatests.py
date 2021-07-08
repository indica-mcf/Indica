from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import pickle

from hda.hdaplot import HDAplot
from hda.hdaworkflow import HDArun

import xarray as xr
from xarray import DataArray
from scipy.optimize import least_squares

plt.ion()

def trust1(hdarun: HDArun):
    """ Conclusion: don't trust SMM!!! """
    print("\n Trust Wp(EFIT), Te(XRCS), Ti(XRCS), test density estimation")
    hdarun.initialize_bckc()
    hdarun.recover_density()

    import matplotlib.pylab as plt

    plt.close("all")
    name = f"{hdarun.data.pulse}_Wp_Te_Ti-test"
    hdarun.plot(name=name, savefig=True)


def trust3(hdarun):
    """ Conclusion: you'd need 2 keV in the centre to get the Wp(EFIT) """
    print("\n Trust Wp(EFIT), Ne(NIR), Ti(XRCS), test Te estimation")
    hdarun.initialize_bckc()
    hdarun.recover_temperature()

    import matplotlib.pylab as plt

    plt.close("all")
    name = f"{hdarun.data.pulse}_Wp_NIR_Ti-test"
    hdarun.plot(name=name, savefig=True)


def test_interferometry(hdarun, interf="smmh1", name="NIRH1_vs_SMMH1", savefig=False):
    """Test effects of different electron density shapes on interferometer measurements"""
    hdarun.initialize_bckc()
    hdarun.bckc.match_interferometer(interf)
    if hasattr(hdarun.bckc, "nirh1"):
        hdarun.bckc.nirh1.values = hdarun.bckc.calc_ne_los_int("nirh1").values
    if hasattr(hdarun.bckc, "smmh1"):
        hdarun.bckc.smmh1.values = hdarun.bckc.calc_ne_los_int("smmh1").values
    standard = deepcopy(hdarun.bckc)

    # H-mode density, L-mode temperature
    ne_0 = hdarun.bckc.profs.ne.sel(rho_poloidal=0).values
    hdarun.bckc.profs.ne = hdarun.bckc.profs.build_density(
        y_0=ne_0,
        y_ped=ne_0 / 1.0,
        x_ped=0.85,
        w_core=0.8,
        w_edge=0.2,
        datatype=("density", "electron"),
    )
    for t in hdarun.bckc.time:
        hdarun.bckc.el_dens.loc[dict(t=t)] = hdarun.bckc.profs.ne.values
    hdarun.bckc.match_interferometer(interf)
    flat_dens = deepcopy(hdarun.bckc)

    HDAplot(flat_dens, standard, name=name, savefig=savefig)


def test_flat_density(hdarun, interf="nirh1"):
    """Trust all measurements, find shape to explain data"""

    # L-mode profiles
    hdarun.initialize_bckc()
    hdarun.match_xrcs()
    hdarun.recover_density()
    standard = deepcopy(hdarun.bckc)

    # H-mode density, L-mode temperature
    ne_0 = hdarun.bckc.profs.ne.sel(rho_poloidal=0).values
    hdarun.bckc.profs.ne = hdarun.bckc.profs.build_density(
        y_0=ne_0,
        y_ped=ne_0 / 1.0,
        x_ped=0.85,
        w_core=0.8,
        w_edge=0.2,
        datatype=("density", "electron"),
    )
    for t in hdarun.bckc.time:
        hdarun.bckc.el_dens.loc[dict(t=t)] = hdarun.bckc.profs.ne.values
    el_dens_int = hdarun.bckc.calc_ne_los_int(interf)
    hdarun.bckc.el_dens *= getattr(hdarun.bckc, interf) / (el_dens_int)

    te_0 = hdarun.bckc.profs.te.sel(rho_poloidal=0).values
    hdarun.bckc.profs.te = hdarun.bckc.profs.build_temperature(
        y_0=te_0,
        y_ped=te_0 / 25.0,
        x_ped=0.95,
        w_core=0.4,
        datatype=("temperature", "electron"),
    )
    hdarun.bckc.profs.te /= hdarun.bckc.profs.te.max()
    for t in hdarun.bckc.time:
        te_0 = hdarun.bckc.el_temp.sel(t=t).sel(rho_poloidal=0).values
        hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values

    for i, elem in enumerate(hdarun.bckc.elements):
        hdarun.bckc.ion_temp.loc[dict(element=elem)] = hdarun.bckc.el_temp.values

    hdarun.match_xrcs()
    hdarun.recover_density()
    flat_dens = deepcopy(hdarun.bckc)

    print((getattr(standard, interf) / getattr(flat_dens, interf)).values)

    HDAplot(flat_dens, standard)


def test_low_edge_temperature(hdarun, zeff=False):

    # low temperature edge
    hdarun.initialize_bckc()
    te_0 = 1.0e3
    hdarun.bckc.profs.te = hdarun.bckc.profs.build_temperature(
        y_0=te_0,
        y_ped=te_0 / 15.0,
        x_ped=0.9,
        w_core=0.2,
        datatype=("temperature", "electron"),
    )
    hdarun.bckc.profs.te /= hdarun.bckc.profs.te.max()
    elements = hdarun.bckc.elements
    main_ion = hdarun.bckc.main_ion
    for t in hdarun.bckc.time:
        te_0 = hdarun.bckc.el_temp.sel(t=t).sel(rho_poloidal=0).values
        hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values
        ti_0 = hdarun.bckc.ion_temp.sel(element=main_ion).sel(t=t).sel(rho_poloidal=0).values
        for elem in elements:
            hdarun.bckc.ion_temp.loc[dict(t=t, element=elem)] = (hdarun.bckc.profs.te * ti_0).values

    hdarun.bckc.match_xrcs()
    hdarun.bckc.simulate_spectrometers()

    # hdarun.recover_zeff(optimize="density")

    hdarun.bckc.propagate_parameters()
    # hdarun.recover_density()

    hdarun.plot()
def rabbit_ears(hdarun: HDArun):

    hdarun.initialize_bckc()
    ne_0 = hdarun.bckc.profs.ne.sel(rho_poloidal=0).values
    hdarun.bckc.profs.ne  = hdarun.bckc.profs.build_density(
        x_0=0.7,
        y_0=ne_0,
        y_ped=ne_0 / 4.0,
        x_ped=0.95,
        w_core=0.1,
        datatype=("density", "electron"),
    )

    for t in hdarun.bckc.time:
        hdarun.bckc.el_dens.loc[dict(t=t)] = hdarun.bckc.profs.ne.values
    hdarun.bckc.match_interferometer(interf)

    # hdarun.recover_density()

    hdarun.plot()

def test_peaked_profiles(hdarun, zeff=False):
    hdarun.initialize_bckc()
    hdarun.recover_density()
    if zeff:
        hdarun.recover_zeff(optimize="density")
    hdarun.bckc.simulate_spectrometers()
    broad = hdarun.bckc

    # Peaked profiles
    hdarun.initialize_bckc()
    te_0 = 1.0e3
    hdarun.bckc.profs.te = hdarun.bckc.profs.build_temperature(
        y_0=te_0,
        y_ped=te_0 / 15.0,
        x_ped=0.9,
        w_core=0.3,
        datatype=("temperature", "electron"),
    )
    hdarun.bckc.profs.te /= hdarun.bckc.profs.te.max()

    ne_0 = 5.0e19
    hdarun.bckc.profs.ne = hdarun.bckc.profs.build_temperature(
        y_0=ne_0,
        y_ped=ne_0 / 15.0,
        x_ped=0.9,
        w_core=0.3,
        datatype=("density", "electron"),
    )
    hdarun.bckc.profs.ne /= hdarun.bckc.profs.ne.max()
    for t in hdarun.bckc.time:
        te_0 = hdarun.bckc.el_temp.sel(t=t).sel(rho_poloidal=0).values
        hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values

        ne_0 = hdarun.bckc.el_dens.sel(t=t).sel(rho_poloidal=0).values
        hdarun.bckc.el_dens.loc[dict(t=t)] = (hdarun.bckc.profs.ne * ne_0).values

    hdarun.bckc.build_current_density()
    hdarun.recover_density()
    if zeff:
        hdarun.recover_zeff(optimize="density")
    hdarun.bckc.simulate_spectrometers()
    peaked = hdarun.bckc

    HDAplot(broad, peaked)


def test_current_density(hdarun):
    """Trust all measurements, find shape to explain data"""

    # L-mode profiles

    # Broad current density
    hdarun.initialize_bckc()
    hdarun.bckc.build_current_density(sigm=0.8)
    hdarun.recover_density()
    hdarun.recover_zeff(optimize="density")
    broad = deepcopy(hdarun.bckc)

    # Peaked current density
    hdarun.initialize_bckc()
    hdarun.bckc.build_current_density(sigm=0.2)
    hdarun.recover_density()
    hdarun.recover_zeff(optimize="density")
    peaked = deepcopy(hdarun.bckc)

    HDAplot(broad, peaked)
