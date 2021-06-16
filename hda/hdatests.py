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

def trust1(hdarun:HDArun):
    """ Conclusion: don't trust SMM!!! """
    print("\n Trust Wp(EFIT), Te(XRCS), Ti(XRCS), test density estimation")
    hdarun.initialize_bckc()
    hdarun.recover_density()

    import matplotlib.pylab as plt

    plt.close("all")
    name = f"{hdarun.data.pulse}_Wp_Te_Ti-test"
    hdarun.plot(name=name, savefig=True)

def trust3(self):
    """ Conclusion: you'd need 2 keV in the centre to get the Wp(EFIT) """
    print("\n Trust Wp(EFIT), Ne(NIR), Ti(XRCS), test Te estimation")
    hdarun.initialize_bckc()
    hdarun.recover_temperature()

    import matplotlib.pylab as plt

    plt.close("all")
    name = f"{hdarun.data.pulse}_Wp_NIR_Ti-test"
    hdarun.plot(name=name, savefig=True)

def test_kinetic_profs(self):
    """Trust all measurements, find shape to explain data"""

    # L-mode profiles
    hdarun.initialize_bckc()
    hdarun.recover_density()
    l_mode = deepcopy(hdarun.bckc)

    # H-mode density, L-mode temperature
    ne_0 = hdarun.bckc.profs.ne.sel(rho_poloidal=0).values
    hdarun.bckc.profs.ne = hdarun.bckc.profs.build_density(
        y_0=ne_0,
        y_ped=ne_0 / 1.5,
        x_ped=0.95,
        w_core=0.9,
        datatype=("density", "electron"),
    )
    hdarun.recover_density()
    h_mode_dens = deepcopy(hdarun.bckc)

    # H-mode density & temperature
    te_0 = hdarun.bckc.profs.te.sel(rho_poloidal=0).values
    hdarun.bckc.profs.te = hdarun.bckc.profs.build_temperature(
        y_0=te_0,
        y_ped=te_0 / 4.0,
        x_ped=0.95,
        w_core=0.6,
        datatype=("temperature", "electron"),
    )
    hdarun.bckc.profs.te /= hdarun.bckc.profs.te.max()
    for t in hdarun.bckc.time:
        te_0 = hdarun.bckc.el_temp.sel(t=t).sel(rho_poloidal=0).values
        hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values
        hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values
    hdarun.recover_density()
    h_mode_both = deepcopy(hdarun.bckc)

    # Hollow temperature
    te_0 = hdarun.bckc.profs.te.sel(rho_poloidal=0).values
    hdarun.bckc.profs.te = hdarun.bckc.profs.build_temperature(
        x_0=0.4,
        y_0=te_0,
        y_ped=te_0 / 4.0,
        x_ped=0.95,
        w_core=0.2,
        datatype=("temperature", "electron"),
    )
    hdarun.bckc.profs.te /= hdarun.bckc.profs.te.max()
    for t in hdarun.bckc.time:
        te_0 = hdarun.bckc.el_temp.sel(t=t).sel(rho_poloidal=0).values
        hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values
    hdarun.recover_density()
    h_mode_hollow = deepcopy(hdarun.bckc)

def test_low_edge_temperature(self):
    hdarun.initialize_bckc()
    hdarun.recover_density()
    hdarun.recover_zeff(optimize="density")
    hdarun.bckc.simulate_spectrometers()
    broad = hdarun.bckc

    # low temperature edge
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
    for t in hdarun.bckc.time:
        te_0 = hdarun.bckc.el_temp.sel(t=t).sel(rho_poloidal=0).values
        hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values

    hdarun.recover_density()
    hdarun.recover_zeff(optimize="density")
    hdarun.bckc.simulate_spectrometers()
    peaked = hdarun.bckc

    HDAplot(broad, peaked)

def test_current_density(self):
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
