from copy import deepcopy

from hda.models.spectrometer import XRCSpectrometer
import hda.physics as ph
from hda.models.plasma import build_data
from hda.models.plasma import Plasma
import hda.plots as plots
import hda.profiles as profiles
from hda.read_st40 import ST40data

from indica.readers import ST40Reader
from indica.equilibrium import Equilibrium

import matplotlib.pylab as plt
import numpy as np
from pytest import approx

plt.ion()

def initialize_plasma_class():
    pulse = 10009
    tstart = 0.015
    tend = 0.125
    dt = 0.01
    st40_reader = ST40Reader(pulse, tstart-0.01, tend+0.01)
    efit_data = st40_reader.get("", "efit", 0)
    equilibrium = Equilibrium(efit_data)

    pl = Plasma(tstart=tstart, tend=tend, dt=dt)
    pl.set_equilibrium(equilibrium)
    pl.set_neutral_density(y1=1.0e15, y0=1.0e9)
    pl.imp_conc.loc[dict(element="c")] = 0.03
    pl.imp_conc.loc[dict(element="ar")] = 0.001
    for t in pl.t:
        pl.el_temp.loc[dict(t=t)] = pl.te_prof.yspl.values
        pl.el_dens.loc[dict(t=t)] = pl.ne_prof.yspl.values
        for elem in pl.impurities:
            pl.imp_dens.loc[dict(t=t, element=elem)] = (
                (pl.el_dens * pl.imp_conc.sel(element=elem)).sel(t=t).values
            )
    return pl

def test_hda():
    pulse = 9780
    tstart = 0.025
    tend = 0.14
    dt = 0.015
    diagn_ne = "smmh1"
    diagn_te = "xrcs"
    quant_ne = "ne"
    quant_te = "te_kw"
    quant_ti = "ti_w"
    quant_ar = "int_w"
    main_ion = "h"
    impurities = ("c", "ar", "he")
    imp_conc = (0.03, 0.001, 0.01)
    savefig = False
    marchuk = True
    extrapolate = None
    name = "standard_hda_test"
    use_ratios = True
    calc_error = False

    # Read ST40 data
    raw = ST40data(pulse, tstart - 0.01, tend + 0.01)
    raw_data = raw.get_all(sxr=False, efit_pulse=None, efit_rev=0)
    bckc = {}
    elements = list(main_ion)
    elements.extend(list(impurities))

    # Initialize plasma class
    pl = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion=main_ion,
        impurities=impurities,
        imp_conc=imp_conc,
    )

    # Build data
    data = build_data(pl, raw_data, equil="efit", pulse=pulse)

    # Assign profiles and initialize data
    profs = profiles.profile_scans(rho=pl.rho)
    pl.Ne_prof = profs["Ne"]["peaked"]
    pl.Te_prof = profs["Te"]["peaked"]
    pl.Ti_prof = profs["Ti"]["peaked"]
    pl.Nimp_prof = profs["Nimp"]["peaked"]
    pl.Vrot_prof = profs["Vrot"]["peaked"]
    for i, elem in enumerate(impurities):
        if elem in pl.imp_conc.element:
            pl.imp_conc.loc[dict(element=elem)] = imp_conc[i]
    pl.set_neutral_density(y1=1.0e15, y0=1.0e9)
    pl.build_atomic_data()
    pl.calculate_geometry()

    if "xrcs" in raw_data:
        pl.forward_models["xrcs"] = XRCSpectrometer(
            marchuk=marchuk, extrapolate=extrapolate
        )

    bckc = pl.match_interferometer(
        data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne
    )
    # TODO: add the impurity concentrations to the __init__ of the plasma class
    pl.calc_imp_dens()

    bckc = pl.match_xrcs_temperatures(
        data,
        bckc=bckc,
        diagnostic=diagn_te,
        quantity_te=quant_te,
        quantity_ti=quant_ti,
        use_ratios=use_ratios,
        calc_error=calc_error,
    )
    bckc = pl.match_xrcs_intensity(
        data, bckc=bckc, diagnostic="xrcs", quantity=quant_ar,
    )
    bckc = pl.interferometer(data, bckc=bckc)
    bckc = pl.bremsstrahlung(data, bckc=bckc)

    return pl, raw_data, data, bckc

    plots.compare_data_bckc(
        data, bckc, raw_data=raw_data, pulse=pl.pulse, savefig=savefig, name=name,
    )
    plots.profiles(pl, data=data, bckc=bckc, savefig=savefig, name=name)
    plots.time_evol(pl, data, bckc=bckc, savefig=savefig, name=name)

    return pl, raw_data, data, bckc


def test_pressure_el():
    pel0 = deepcopy(PL.pressure_el)
    _pel0 = ph.calc_pressure(PL.el_dens, PL.el_temp)
    PL.el_dens *= 1.01
    pel1 = deepcopy(PL.pressure_el)
    PL.el_dens *= 0.9
    pel2 = deepcopy(PL.pressure_el)

    assert np.all(pel0.values == approx(_pel0.values))
    assert np.all(pel1 > pel0)
    assert np.all(pel2 < pel1)
    assert np.all(pel2 < pel0)

    return PL


def test_sxr_rad():
    pel0 = deepcopy(PL.pressure_el)
    _pel0 = ph.calc_pressure(PL.el_dens, PL.el_temp)
    PL.el_dens *= 1.01
    pel1 = deepcopy(PL.pressure_el)
    PL.el_dens *= 0.9
    pel2 = deepcopy(PL.pressure_el)

    assert np.all(pel0.values == approx(_pel0.values))
    assert np.all(pel1 > pel0)
    assert np.all(pel2 < pel1)
    assert np.all(pel2 < pel0)

    return PL


def test_fz():
    fz = deepcopy(PL.fz)
    # _pel0 = ph.calc_pressure(PL.el_dens, PL.el_temp)
    # PL.el_dens *= 1.01
    # pel1 = deepcopy(PL.pressure_el)
    # PL.el_dens *= 0.9
    # pel2 = deepcopy(PL.pressure_el)
    #
    # assert np.all(pel0.values == approx(_pel0.values))
    # assert np.all(pel1 > pel0)
    # assert np.all(pel2 < pel1)
    # assert np.all(pel2 < pel0)
    return fz
