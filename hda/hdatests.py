from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import pickle

from hda.hdaplot import HDAplot
from hda.hdaworkflow import HDArun
from hda.spline_profiles import Plasma_profs
from hda.read_st40 import ST40data
from hda.plasma import Plasma
import hda.plots as plots
import hda.hda_tree as hda_tree
from indica.readers import ST40Reader

from hda.diagnostics.spectrometer import XRCSpectrometer
from hda.diagnostics.PISpectrometer import PISpectrometer

plt.ion()

def compare_astra(pulse=8574, tstart=0.02, tend=0.14, revision=105, interf="nirh1"):
    pulse=8574
    tstart=0.02
    tend=0.14
    revision=105
    reader = ST40Reader(int(pulse+25.e6), tstart, tend, tree="astra")
    astra = reader.get("", "astra", revision)
    astra["ne"] *= 1.e19
    astra["ni"] *= 1.e19
    astra["te"] *= 1.e3
    astra["ti"] *= 1.e3
    # rho_poloidal = astra["p"].rho_poloidal
    # rho_toroidal = astra["ne"].rho_toroidal

    time = astra["te"].t
    tstart, tend = time[0], time[-1]
    dt = time[1] - time[0]
    pl, raw_data, data, bckc = plasma_workflow(pulse=pulse, tstart=tstart, tend=tend, dt=dt)

def plasma_workflow(
    pulse=9229,
    tstart=0.025,
    tend=0.14,
    dt=0.015,
    diagn_ne="smmh1",
    write=False,
    modelling=True,
    recover_dens=False,
):
    """
    New framework for running HDA

    Pulses analysed up to now
    NBI [8338, 8373, 8374, 8574, 8575, 8582, 8583, 8597, 8598, 8599, 9184, 9219, 9221, 9229]
        res = tests.plasma_workflow(pulse=8374, tstart=0.02, tend=0.17, dt=0.01, diagn_ne="nirh1_bin", write=True)
        res = tests.plasma_workflow(pulse=9219, tstart=0.02, tend=0.1, dt=0.007, diagn_ne="nirh1_bin", write=True)
        res = tests.plasma_workflow(pulse=9221, tstart=0.02, tend=0.1, dt=0.007, diagn_ne="nirh1_bin", write=True)
        res = tests.plasma_workflow(pulse=9229, tstart=0.025, tend=0.12, dt=0.015, diagn_ne="smmh1", write=True)

    Ohmic [8385, 8386, 8387, 8390, 8405, 8458, 8909, 9184]
        res = tests.plasma_workflow(pulse=8387, tstart=0.035, tend=0.12, dt=0.015, diagn_ne="nirh1_bin", recover_dens=True, write=True)
        res = tests.plasma_workflow(pulse=8458, tstart=0.025, tend=0.12, dt=0.015, diagn_ne="nirh1_bin", recover_dens=True, write=True)
        res = tests.plasma_workflow(pulse=8909, tstart=0.025, tend=0.13, dt=0.01, diagn_ne="nirh1_bin", write=True)
        res = tests.plasma_workflow(pulse=9184, tstart=0.025, tend=0.118, dt=0.015, diagn_ne="nirh1_bin", recover_dens=True, write=True)

    """

    quant_ne = "ne"
    diagn_te = "xrcs"
    quant_te = "te_kw"
    quant_ti = "ti_w"

    # Read raw data
    raw = ST40data(pulse, tstart-0.02, tend+0.02)
    raw.get_all()  # smmh1_rev=1)
    raw_data = raw.data

    # Plasma class
    bckc = {}
    pl = Plasma(tstart=tstart, tend=tend, dt=dt)
    data = pl.build_data(raw_data, pulse=pulse)

    # Impose impurity concentration and calculate dilution
    imp_conc = {"c": 0.03, "ar": 0.0005}
    for elem in imp_conc:
        if elem in pl.ion_conc.element:
            pl.ion_conc.loc[dict(element=elem)] = imp_conc[elem]

    pl.build_atomic_data()
    pl.calculate_geometry()
    if "xrcs" in raw_data:
        pl.forward_models["xrcs"] = XRCSpectrometer()
    if "princeton" in raw_data:
        pl.forward_models["princeton"] = PISpectrometer()

    # Rescale density to match interferometer
    bckc = pl.match_interferometer(
        data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne
    )

    # Build temperature profiles to match XRCS
    bckc = pl.match_xrcs(
        data,
        bckc=bckc,
        diagnostic=diagn_te,
        quantity_te=quant_te,
        quantity_ti=quant_ti,
    )

    # Go through the rest of the computation
    pl.calc_meanz()
    pl.calc_imp_dens()
    pl.calc_main_ion_dens()
    pl.calc_zeff()
    bckc = pl.calc_pressure(data=data, bckc=bckc)

    # Recover density to match stored energy
    if recover_dens:
        pl.recover_density(data)

        # Back-calculate all diagnostic measurements
        pl.calc_meanz()
        pl.calc_imp_dens()
        pl.calc_main_ion_dens()
        pl.calc_zeff()
        bckc = pl.calc_pressure(data=data, bckc=bckc)

    bckc = pl.interferometer(data, bckc=bckc)

    # Compare diagnostic data with back-calculated data
    plots.compare_data_bckc(data, bckc, raw_data=raw_data, pulse=pl.pulse)
    plots.profiles(pl)

    run_name = "RUN40"
    descr = f"New profile shapes and ionisation balance"
    if write:
        if modelling:
            pulse = pl.pulse + 25000000
        else:
            pulse = pl.pulse
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name
        )

    return pl, raw_data, data, bckc


def best_astra(
    pulse=8383, tstart=0.02, tend=0.12, hdarun=None, write=False, force=False
):
    """
    Best profile shapes from ASTRA runs of 8383 applied to database
    """
    ohmic_pulses = [8385, 8386, 8387, 8390, 8405, 8458]  # 8401
    nbi_pulses = [8338, 8373, 8374, 8574, 8575, 8582, 8583, 8597, 8598, 8599]  #

    pulses = np.sort(np.concatenate((np.array(ohmic_pulses), np.array(nbi_pulses))))
    if pulse is not None:
        pulses = [pulse]
    for pulse in pulses:
        interf = "nirh1"
        hdarun = HDArun(pulse=pulse, interf=interf, tstart=tstart, tend=tend)

        # Rebuild temperature profiles
        hdarun.profiles_nbi()
        profs_spl = Plasma_profs(hdarun.data.time)

        # Rescale to match XRCS measurements
        hdarun.data.match_xrcs(profs_spl=profs_spl)

        # Recalculate average charge, dilution, Zeff, total pressure
        hdarun.data.calc_meanz()
        hdarun.data.calc_main_ion_dens(fast_dens=False)
        hdarun.data.impose_flat_zeff()
        hdarun.data.calc_main_ion_dens(fast_dens=False)
        hdarun.data.calc_zeff()
        hdarun.data.calc_pressure()

        descr = f"Best profile shapes from ASTRA {pulse}, c_C=3%"
        run_name = "RUN30"
        plt.close("all")
        hdarun.plot()
        if write:
            hdarun.write(hdarun.data, descr=descr, run_name=run_name, force=force)
        else:
            return hdarun


def scan_profile_shape(pulse=8383, hdarun=None, write=False):
    """
    Fix edge plasma parameters (rho > 0.8) and scan profile shapes
    """

    interf = "nirh1"
    if hdarun is None:
        hdarun = HDArun(pulse=pulse, interf=interf, tstart=0.02, tend=0.1)

    # Temperature profile shape scan, flat density
    hdarun.profiles_ohmic()

    te_flat = deepcopy(hdarun)
    te_peak1 = deepcopy(hdarun)
    te_peak2 = deepcopy(hdarun)

    profs_spl = Plasma_profs(te_flat.data.time)
    te_flat.data.match_xrcs(profs_spl=profs_spl)
    te_flat.data.calc_pressure()
    descr = "Flat density, flat temperature, c_C=3%"
    run_name = "RUN10"
    if write == True:
        te_flat.write(te_flat.data, descr=descr, run_name=run_name)

    profs_spl.el_temp.scale(2.0, dim_lim=(0, 0))
    te_peak1.data.match_xrcs(profs_spl=profs_spl)
    te_peak1.data.calc_pressure()
    descr = "Flat density, peaked temperature, c_C=3%"
    run_name = "RUN11"
    if write == True:
        te_peak1.write(te_peak1.data, descr=descr, run_name=run_name)

    profs_spl.el_temp.scale(0.5, dim_lim=(0.7, 0.98))
    profs_spl.ion_temp.scale(2.0, dim_lim=(0, 0))
    profs_spl.ion_temp.scale(0.5, dim_lim=(0.7, 0.98))
    te_peak2.data.match_xrcs(profs_spl=profs_spl)
    te_peak2.data.calc_pressure()
    descr = "Flat density, very peaked temperature, c_C=3%"
    run_name = "RUN12"
    if write == True:
        te_peak2.write(te_peak2.data, descr=descr, run_name=run_name)

    flat_dens = {"te_flat": te_flat, "te_peak1": te_peak1, "te_peak2": te_peak2}

    # Peaked density
    hdarun.profiles_nbi()

    te_flat = deepcopy(hdarun)
    te_peak1 = deepcopy(hdarun)
    te_peak2 = deepcopy(hdarun)

    profs_spl = Plasma_profs(te_flat.data.time)
    te_flat.data.match_xrcs(profs_spl=profs_spl)
    te_flat.data.calc_pressure()
    descr = "Peaked density, flat temperature, c_C=3%"
    run_name = "RUN20"
    if write == True:
        te_flat.write(te_flat.data, descr=descr, run_name=run_name)

    profs_spl.el_temp.scale(2.0, dim_lim=(0, 0))
    te_peak1.data.match_xrcs(profs_spl=profs_spl)
    te_peak1.data.calc_pressure()
    descr = "Peaked density, peaked temperature, c_C=3%"
    run_name = "RUN21"
    if write == True:
        te_peak1.write(te_peak1.data, descr=descr, run_name=run_name)

    profs_spl.el_temp.scale(0.5, dim_lim=(0.7, 0.98))
    profs_spl.ion_temp.scale(2.0, dim_lim=(0, 0))
    profs_spl.ion_temp.scale(0.5, dim_lim=(0.7, 0.98))
    te_peak2.data.match_xrcs(profs_spl=profs_spl)
    te_peak2.data.calc_pressure()
    descr = "Peaked density, very peaked temperature, c_C=3%"
    run_name = "RUN22"
    if write == True:
        te_peak2.write(te_peak2.data, descr=descr, run_name=run_name)

    peaked_dens = {"te_flat": te_flat, "te_peak1": te_peak1, "te_peak2": te_peak2}

    if not write:
        return flat_dens, peaked_dens


def ohmic_pulses(write=False, interf="smmh1", match_kinetic=False):
    # pulses = [8385, 8386, 8387, 8390, 8401, 8405, 8458]
    for pulse in pulses:
        hdarun = HDArun(pulse=pulse, interf=interf, tstart=0.02, tend=0.1)
        hdarun.profiles_ohmic()
        if match_kinetic:
            hdarun.data.calc_pressure()
            descr = "New profile shapes, match kinetic profiles only, c_C=3%"
            run_name = "RUN01"
        else:
            hdarun.match_energy()
            descr = "New profile shapes, adapt Ne to match Wmhd, c_C=3%"
            run_name = "RUN05"
        if write == True:
            hdarun.write(hdarun.bckc, descr=descr, run_name=run_name)
        else:
            hdarun.plot()

    return hdarun


def NBI_pulses(write=False, interf="smmh1", match_kinetic=False):
    pulses = [8338, 8574, 8575, 8582, 8583, 8597, 8598, 8599]
    interf = ["nirh1"] * len(pulses)
    for i, pulse in enumerate(pulses):
        plt.close("all")
        hdarun = HDArun(
            pulse=pulse, interf=interf[i], tstart=0.015, tend=0.14, dt=0.015
        )
        hdarun.profiles_nbi()
        if match_kinetic:
            hdarun.data.calc_pressure()
            descr = "New profile shapes, match kinetic measurements only, c_C=3%"
            run_name = "RUN01"
        else:
            hdarun.match_energy()
            descr = "New profile shapes, adapt Ne to match Wmhd, c_C=3%"
            run_name = "RUN05"
        if write == True:
            hdarun.write(hdarun.data, descr=descr, run_name=run_name)
        else:
            hdarun.plot()
            _ = input("press...")

    return hdarun


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
        ti_0 = (
            hdarun.bckc.ion_temp.sel(element=main_ion)
            .sel(t=t)
            .sel(rho_poloidal=0)
            .values
        )
        for elem in elements:
            hdarun.bckc.ion_temp.loc[dict(t=t, element=elem)] = (
                hdarun.bckc.profs.te * ti_0
            ).values

    hdarun.bckc.match_xrcs()
    hdarun.bckc.simulate_spectrometers()

    # hdarun.recover_zeff(optimize="density")

    hdarun.bckc.propagate_parameters()
    # hdarun.recover_density()

    hdarun.plot()


def rabbit_ears(hdarun: HDArun):

    hdarun.initialize_bckc()
    ne_0 = hdarun.bckc.profs.ne.sel(rho_poloidal=0).values
    hdarun.bckc.profs.ne = hdarun.bckc.profs.build_density(
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
