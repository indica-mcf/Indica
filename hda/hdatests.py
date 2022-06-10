from copy import deepcopy
import os
import pickle

from hda.diagnostics.PISpectrometer import PISpectrometer
from hda.diagnostics.spectrometer import XRCSpectrometer
import hda.hda_tree as hda_tree
from hda.plasma import initialize_bckc
from hda.plasma import Plasma
from hda.plasma import remap_diagnostic
import hda.plots as plots
import hda.profiles as profiles
from hda.read_st40 import ST40data
import matplotlib.pylab as plt
import numpy as np
from scipy import constants
import xarray as xr
from xarray import DataArray

from indica.converters.time import bin_in_time_dt
from indica.operators.atomic_data import PowerLoss
from indica.readers import ADASReader
from indica.readers import ST40Reader

plt.ion()

"""
descr = "Line ratio analysis n3/(n3+n4+n5+w), same profile shapes as RUN40"
run_name = "RUN50"
pulses = [9229, 9391, 9539]
write = False
save_pickle = True
for pulse in pulses:
    res = tests.plasma_workflow(pulse=pulse, tstart=0.02, tend=0.12, dt=0.007,
        diagn_ne="smmh1", quant_te="te_n3w", imp_conc=(0.03, 0.001, 0.01), marchuk=True,
        use_ratios=True, xrcs_time=False, descr=descr, run_name=run_name, calc_error=True,
        write=write, save_pickle=save_pickle)

"""


def test_hda(
    pulse=9780,
    tstart=0.02,
    tend=0.14,
    dt=0.02,
    diagn_ne="smmh1",
    diagn_te="xrcs",
    quant_ne="ne",
    quant_te="te_kw",
    quant_ti="ti_w",
    quant_ar="int_w",
    main_ion="h",
    impurities=("c", "ar", "he"),
    imp_conc=(0.03, 0.001, 0.01),
    cal_ar=1.0,
    savefig=False,
    ne_peaking=None,
    marchuk=True,
    extrapolate=None,
    name="standard_hda_test",
    xrcs_time=False,
    use_ratios=True,
    calc_error=False,
    sxr=False,
    efit_pulse=None,
    efit_run=0,
):

    raw = ST40data(pulse, tstart - 0.01, tend + 0.01)
    raw.get_all(sxr=sxr, efit_pulse=efit_pulse, efit_rev=efit_run)  # smmh1_rev=2)
    raw_data = raw.data
    dt_xrcs = (raw_data["xrcs"]["ti_w"].t[1] - raw_data["xrcs"]["ti_w"].t[0]).values
    if xrcs_time and ("xrcs" in raw_data.keys()):
        time = raw_data["xrcs"]["ti_w"].t.values
        tind = np.argwhere((time >= tstart) * (time <= tend)).flatten()
        tstart = time[tind[0]]
        tend = time[tind[-1]]
        dt = dt_xrcs
    bckc = {}
    elements = list(main_ion)
    elements.extend(list(impurities))

    pl = Plasma(tstart=tstart, tend=tend, dt=dt, elements=elements)
    data = pl.build_data(raw_data, pulse=pulse)
    data = pl.apply_limits(data, "xrcs", err_lim=(np.nan, np.nan))

    profs = profiles.profile_scans(rho=pl.rho)
    pl.Ne_prof = profs["Ne"]["peaked"]
    pl.Te_prof = profs["Te"]["peaked"]
    pl.Ti_prof = profs["Ti"]["peaked"]
    pl.Nimp_prof = profs["Nimp"]["peaked"]
    pl.Vrot_prof = profs["Vrot"]["peaked"]

    if ne_peaking is not None:
        pl.Ne_prof.peaking = ne_peaking
        pl.Ne_prof.build_profile()
    for i, elem in enumerate(impurities):
        if elem in pl.ion_conc.element:
            pl.ion_conc.loc[dict(element=elem)] = imp_conc[i]
    pl.set_neutral_density(y1=1.0e15, y0=1.0e9)
    pl.build_atomic_data()
    pl.calculate_geometry()
    if "xrcs" in raw_data:
        pl.forward_models["xrcs"] = XRCSpectrometer(
            marchuk=marchuk, extrapolate=extrapolate
        )
    if "princeton" in raw_data:
        pl.forward_models["princeton"] = PISpectrometer()

    bckc = pl.match_interferometer(
        data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne
    )
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
    # pl.el_dens.plot()
    # pl.calc_fz_lz()
    # pl.calc_meanz()
    pl.calc_imp_dens()
    bckc = pl.match_xrcs_intensity(
        data,
        bckc=bckc,
        diagnostic="xrcs",
        quantity=quant_ar,
        cal=cal_ar,
        dt=dt_xrcs,
    )
    # pl.calc_main_ion_dens()
    # pl.calc_zeff()
    pl.calc_pressure()
    pl.calc_rad_power()
    pl.map_to_midplane()
    # pl.map_to_rova()
    bckc = pl.interferometer(data, bckc=bckc)
    bckc = pl.bremsstrahlung(data, bckc=bckc)

    plots.compare_data_bckc(
        data,
        bckc,
        raw_data=raw_data,
        pulse=pl.pulse,
        savefig=savefig,
        name=name,
    )
    plots.profiles(pl, data=data, bckc=bckc, savefig=savefig, name=name)
    plots.time_evol(pl, data, bckc=bckc, savefig=savefig, name=name)

    return pl, raw_data, data, bckc


def plasma_workflow(
    pulse=9229,
    tstart=0.025,
    tend=0.14,
    dt=0.015,
    diagn_ne="smmh1",
    diagn_te="xrcs",
    quant_ne="ne",
    quant_te="te_kw",
    quant_ti="ti_w",
    quant_ar="int_w",
    main_ion="h",
    impurities=("c", "ar", "he"),
    imp_conc=(0.03, 0.001, 0.01),
    cal_ar=1.0,
    write=False,
    save_pickle=False,
    savefig=False,
    plotfig=False,
    modelling=True,
    recover_dens=False,
    ne_peaking=None,
    marchuk=True,
    extrapolate=None,
    run_name="RUN40",
    descr="New profile shapes and ionisation balance",
    name="",
    xrcs_time=True,
    use_ratios=True,
    calc_error=False,
    sxr=False,
    efit_pulse=None,
    efit_run=0,
):
    """
    New framework for running HDA

    Pulses analysed up to now
    Jari's beam tests
        res = tests.plasma_workflow(pulse=9188, tstart=0.02, tend=0.17, dt=0.007, diagn_ne="nirh1_bin") #not good...
        res = tests.plasma_workflow(pulse=9189, tstart=0.02, tend=0.17, dt=0.010, diagn_ne="nirh1_bin") #problem with Te
        res = tests.plasma_workflow(pulse=9190, tstart=0.02, tend=0.17, dt=0.007, diagn_ne="nirh1_bin") #not good...
        res = tests.plasma_workflow(pulse=9191, tstart=0.02, tend=0.17, dt=0.007, diagn_ne="nirh1_bin") #not good...

    NBI [8338, 8373, 8374, 8574, 8575, 8582, 8583, 8597, 8598, 8599, 9184, 9219, 9221, 9229]
        res = tests.plasma_workflow(pulse=8599, tstart=0.02, tend=0.17, dt=0.01, diagn_ne="nirh1_bin")
        res = tests.plasma_workflow(pulse=9219, tstart=0.02, tend=0.1, dt=0.007, diagn_ne="nirh1_bin")
        res = tests.plasma_workflow(pulse=9221, tstart=0.02, tend=0.1, dt=0.007, diagn_ne="nirh1_bin")
        res = tests.plasma_workflow(pulse=9229, tstart=0.02, tend=0.12, dt=0.007, diagn_ne="smmh1", quant_te="te_n3w")

    Ohmic [8385, 8386, 8387, 8390, 8405, 8458, 8909, 9184]
        res = tests.plasma_workflow(pulse=8387, tstart=0.035, tend=0.17, dt=0.015, diagn_ne="nirh1_bin", recover_dens=True)
        res = tests.plasma_workflow(pulse=8458, tstart=0.025, tend=0.12, dt=0.015, diagn_ne="nirh1_bin", recover_dens=True)
        res = tests.plasma_workflow(pulse=8909, tstart=0.025, tend=0.13, dt=0.01, diagn_ne="nirh1_bin")
        res = tests.plasma_workflow(pulse=9184, tstart=0.025, tend=0.2, dt=0.015, diagn_ne="nirh1_bin", recover_dens=True)

    Bremsstrahlung & Ar concentration calculation
        res = tests.plasma_workflow(pulse=9408, tstart=0.02, tend=0.11, dt=0.007, diagn_ne="smmh1", ne_peaking=1, quant_te="te_n3w")

    """

    if write:
        save_pickle = True
        savefig = True
        name = deepcopy(run_name)

    # Read raw data
    raw = ST40data(pulse, tstart - 0.01, tend + 0.01)
    raw.get_all(sxr=sxr, efit_pulse=efit_pulse, efit_rev=efit_run)  # smmh1_rev=2)
    raw_data = raw.data
    dt_xrcs = (raw_data["xrcs"]["ti_w"].t[1] - raw_data["xrcs"]["ti_w"].t[0]).values
    if "xrcs" in raw_data.keys() and xrcs_time:
        time = raw_data["xrcs"]["ti_w"].t.values
        tind = np.argwhere((time >= tstart) * (time <= tend)).flatten()
        tstart = time[tind[0]]
        tend = time[tind[-1]]
        dt = dt_xrcs

    # Plasma class
    bckc = {}
    elements = list(main_ion)
    elements.extend(list(impurities))

    pl = Plasma(tstart=tstart, tend=tend, dt=dt, elements=elements)
    data = pl.build_data(raw_data, pulse=pulse)
    data = pl.apply_limits(data, "xrcs", err_lim=(np.nan, np.nan))

    if ne_peaking is not None:
        pl.Ne_prof.peaking = ne_peaking
        pl.Ne_prof.build_profile()

    # Impose impurity concentration and calculate dilution
    for i, elem in enumerate(impurities):
        if elem in pl.ion_conc.element:
            pl.ion_conc.loc[dict(element=elem)] = imp_conc[i]
    pl.set_neutral_density(y1=1.0e15, y0=1.0e9)
    pl.build_atomic_data()
    pl.calculate_geometry()
    if "xrcs" in raw_data:
        pl.forward_models["xrcs"] = XRCSpectrometer(
            marchuk=marchuk, extrapolate=extrapolate
        )
    if "princeton" in raw_data:
        pl.forward_models["princeton"] = PISpectrometer()

    # Rescale density to match interferometer
    bckc = pl.match_interferometer(
        data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne
    )

    # Default impurity concentrations
    pl.calc_imp_dens()
    #
    # return pl, raw_data, data, bckc

    # Build temperature profiles to match XRCS
    bckc = pl.match_xrcs_temperatures(
        data,
        bckc=bckc,
        diagnostic=diagn_te,
        quantity_te=quant_te,
        quantity_ti=quant_ti,
        use_ratios=use_ratios,
        calc_error=calc_error,
    )

    # return pl, raw_data, data, bckc

    # Average charge known Te
    pl.el_dens.plot()
    pl.calc_fz_lz()
    pl.calc_meanz()

    # Default impurity concentrations
    pl.calc_imp_dens()

    # Ar density from intensity of w line
    bckc = pl.match_xrcs_intensity(
        data,
        bckc=bckc,
        diagnostic="xrcs",
        quantity=quant_ar,
        cal=cal_ar,
        dt=dt_xrcs,
    )
    # Quasineutrality
    pl.calc_main_ion_dens()
    pl.calc_zeff()
    pl.calc_pressure()

    # Recover density to match stored energy
    # if recover_dens:
    #     pl.recover_density(data)
    #     pl.calc_meanz()
    #     pl.calc_imp_dens()
    #     bckc = pl.match_xrcs_intensity(
    #         data, bckc=bckc, diagnostic="xrcs", quantity=quant_ar, mult=mult_ar,
    #     )
    #     pl.calc_main_ion_dens()
    #     pl.calc_zeff()
    #     bckc = pl.calc_pressure()

    pl.calc_rad_power()

    # pl.build_current_density()
    # pl.calc_magnetic_field()
    # pl.calc_beta_poloidal()
    # pl.calc_vloop()

    bckc = pl.interferometer(data, bckc=bckc)
    bckc = pl.bremsstrahlung(data, bckc=bckc)

    # return pl, raw_data, data, bckc

    # Compare diagnostic data with back-calculated data
    if plotfig or savefig:
        plots.compare_data_bckc(
            data,
            bckc,
            raw_data=raw_data,
            pulse=pl.pulse,
            savefig=savefig,
            name=name,
        )
        plots.profiles(pl, data=data, savefig=savefig, name=name)
        plots.time_evol(pl, data, bckc=bckc, savefig=savefig, name=name)

    if write:
        if modelling:
            pulse = pl.pulse + 25000000
        else:
            pulse = pl.pulse
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name
        )

    if save_pickle or write:
        save_to_pickle(pl, raw_data, data, bckc, pulse=pl.pulse, name=run_name)

    return pl, raw_data, data, bckc


def save_to_pickle(pl, raw_data, data, bckc, pulse=0, name="", force=False):
    picklefile = f"/home/marco.sertoli/data/Indica/{pulse}_{name}_HDA.pkl"

    if (
        os.path.isfile(picklefile)
        and not (os.access(picklefile, os.W_OK))
        and force is True
    ):
        os.chmod(picklefile, 0o744)

    if os.access(picklefile, os.W_OK) or not os.path.isfile(picklefile):
        pickle.dump([pl, raw_data, data, bckc], open(picklefile, "wb"))
        os.chmod(picklefile, 0o444)


def load_pickle(pulse, name):
    picklefile = f"/home/marco.sertoli/data/Indica/{pulse}_{name}_HDA.pkl"
    return pickle.load(open(picklefile, "rb"))


def plot_results(pl, raw_data, data, bckc, savefig=False, name=""):
    if savefig:
        plt.ioff()
    plots.compare_data_bckc(
        data,
        bckc,
        raw_data=raw_data,
        pulse=pl.pulse,
        savefig=savefig,
        name=name,
    )
    plots.profiles(pl, data=data, savefig=savefig, name=name)
    plots.time_evol(pl, data, bckc=bckc, savefig=savefig, name=name)
    if savefig:
        plt.ion()


def propagate(pl, raw_data, data, bckc, quant_ar="int_w", cal_ar=1):
    dt_xrcs = (raw_data["xrcs"]["ti_w"].t[1] - raw_data["xrcs"]["ti_w"].t[0]).values
    pl.calc_meanz()
    pl.calc_imp_dens()
    pl.match_xrcs_intensity(
        data,
        bckc=bckc,
        diagnostic="xrcs",
        quantity=quant_ar,
        cal=cal_ar,
        dt=dt_xrcs,
    )
    pl.calc_main_ion_dens()
    pl.calc_zeff()
    pl.calc_pressure()
    pl.calc_rad_power()
    pl.interferometer(data, bckc=bckc)
    pl.bremsstrahlung(data, bckc=bckc)
    pl.map_to_midplane()

    return pl, bckc


def run_all_scans(efit_pulse=None, efit_run=None, run_add="", force=True):
    # pulses = [8532, 8533, 8605, 8621, 9098, 9099, 9229, 9401, 9486, 9537, 9538, 9539, 9619, 9622,
    # 9624, 9626, 9676, 9721, 9746, 9748, 9752, 9766, 9771, 9779, 9780, 9781, 9783, 9784, 9787, 9816,
    # 9822, 9823, 9824, 9831, 9835, 9837, 9839, 9840, 9841, 9842, 9849, 9877, 9878, 9880, 9885, 9892,
    # 9894, 9896, 9901, 9913, 9928, 10014]

    # 9818, 9820, 9389 - unknown issues
    # 9840 - doesn't have enough Ar
    # 9623 - issues with XRCS temperature optimisation...
    # 10013 - issues with EFIT

    # pulses = [9850] * 2
    # efit_pulse = [11009850] * 2
    # efit_run = ["1016A2", "1013N"]

    pulses = [10014]
    tlims = [(0.02, 0.10)] * len(pulses)
    run_add = ["MID"]*len(pulses)
    efit_pulse = [efit_pulse]*len(pulses)
    efit_run = [0]*len(pulses)
    only_run = None  # :int = write only this run

    for pulse, tlim, _efit_pulse, _efit_run, _run_add in zip(pulses, tlims, efit_pulse, efit_run, run_add):
        print(pulse)
        scan_profiles(
            pulse,
            tstart=tlim[0],
            tend=tlim[1],
            dt=0.01,
            diagn_ne="smmh1",
            quant_ne="ne",
            quant_te="te_n3w",
            quant_ti="ti_w",
            c_c=0.03,
            c_ar=0.001,
            c_he=0.01,
            marchuk=True,
            extrapolate=None,
            write=True,
            save_pickle=True,
            plotfig=False,
            savefig=False,
            modelling=True,
            xrcs_time=False,
            force=force,
            sxr=False,
            main_ion="h",
            proceed=True,
            efit_run=_efit_run,
            efit_pulse=_efit_pulse,
            only_run=only_run,
            run_add=_run_add,
        )


def scan_profiles(
    pulse=9229,
    tstart=0.02,
    tend=0.12,
    dt=0.01,
    diagn_ne="smmh1",
    quant_ne="ne",
    quant_te="te_n3w",
    quant_ti="ti_w",
    c_c=0.03,
    c_ar=0.001,
    c_he=0.01,
    marchuk=True,
    extrapolate=None,
    write=False,
    save_pickle=False,
    savefig=False,
    plotfig=False,
    modelling=True,
    xrcs_time=True,
    res=None,
    force=True,
    sxr=False,
    main_ion="h",
    proceed=True,
    run_add="",
    efit_run=0,
    efit_pulse=None,
    only_run=None,
):
    print("Scanning combinations of profile shapes")

    # TODO: create better calculation of all average quantities and uncertainties...

    profs = profiles.profile_scans()

    if res is None:
        res = plasma_workflow(
            pulse=pulse,
            tstart=tstart,
            tend=tend,
            dt=dt,
            diagn_ne=diagn_ne,
            quant_ne=quant_ne,
            diagn_te="xrcs",
            quant_te=quant_te,
            imp_conc=(c_c, c_ar, c_he),
            marchuk=marchuk,
            extrapolate=extrapolate,
            xrcs_time=xrcs_time,
            use_ratios=True,
            calc_error=False,
            cal_ar=1,
            sxr=sxr,
            main_ion=main_ion,
            plotfig=False,
            efit_pulse=efit_pulse,
            efit_run=efit_run,
        )
        if not proceed:
            return res

    pl, raw_data, data, bckc = res

    pulse = pl.pulse
    if modelling:
        pulse_to_write = pulse + 25000000
    else:
        pulse_to_write = pulse

    # pl.Vrot_prof = prof_list["Vrot"][0]
    run = 60
    run_tmp = deepcopy(run)
    pl_dict = {}
    bckc_dict = {}
    run_dict = {}
    iteration = 0
    for kNe, Ne in profs["Ne"].items():
        pl.Ne_prof = deepcopy(Ne)
        for kTe, Te in profs["Te"].items():
            pl.Te_prof = deepcopy(Te)
            for kTi, Ti in profs["Ti"].items():
                Ti = deepcopy(Te)
                Ti.datatype = ("temperature", "ion")
                pl.Ti_prof = Ti
                if kTi == "broad":
                    use_ref = False
                else:
                    use_ref = True
                for kNimp, Nimp in profs["Nimp"].items():
                    if kNimp != "peaked":
                        Nimp = deepcopy(Ne)
                        Nimp.datatype = ("density", "impurity")
                        Nimp.y1 = (
                            Nimp.yspl.sel(rho_poloidal=0.7, method="nearest").values
                            / 1.5
                        )
                        Nimp.yend = Nimp.y1
                        Nimp.build_profile()
                    pl.Nimp_prof = deepcopy(Nimp)

                    run_tmp += 1
                    if only_run is not None:
                        if run_tmp != only_run:
                            continue

                    run_name = f"RUN{run_tmp}{run_add}"
                    descr = f"{kTe} Te, {kTi} Ti, {kNe} Ne, {kNimp} Cimp"
                    print(f"\n{descr}\n")
                    run_dict[run_name] = descr

                    pl.match_interferometer(
                        data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne
                    )
                    pl.calc_imp_dens()

                    pl.match_xrcs_temperatures(
                        data,
                        bckc=bckc,
                        diagnostic="xrcs",
                        quantity_te=quant_te,
                        quantity_ti=quant_ti,
                        use_ratios=True,
                        calc_error=False,
                        use_ref=use_ref,
                    )
                    propagate(pl, raw_data, data, bckc, quant_ar="int_w")

                    pl_dict[run_name] = deepcopy(pl)
                    bckc_dict[run_name] = deepcopy(bckc)

                    if plotfig or savefig:
                        plot_results(
                            pl, raw_data, data, bckc, savefig=savefig, name=run_name
                        )
                        if not savefig:
                            input("Press any key to continue")

                    if write:
                        hda_tree.write(
                            pl,
                            pulse_to_write,
                            "HDA",
                            data=data,
                            bckc=bckc,
                            descr=descr,
                            run_name=run_name,
                            force=force,
                        )

                    if save_pickle or write:
                        save_to_pickle(
                            pl,
                            raw_data,
                            data,
                            bckc,
                            pulse=pl.pulse,
                            name=run_name,
                            force=force,
                        )

                    if run_tmp > 80:
                        break

                    iteration += 1

    elem = "ar"
    # t = pl.time.values[int(len(pl.time) / 2.0)]

    ne0, ni0, nimp0, te0, ti0 = [], [], [], [], []
    el_dens, ion_dens, neutral_dens, el_temp, ion_temp, meanz, zeff, pressure_th = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    runs = []
    for run_name, pl in pl_dict.items():
        runs.append(run_name)
        el_dens.append(pl.el_dens)
        ion_dens.append(pl.ion_dens)
        neutral_dens.append(pl.neutral_dens)
        el_temp.append(pl.el_temp)
        ion_temp.append(pl.ion_temp)
        meanz.append(pl.meanz)
        zeff.append(pl.zeff)
        pressure_th.append(pl.pressure_th)
        ne0.append(pl.el_dens.sel(rho_poloidal=0))
        ni0.append(pl.ion_dens.sel(rho_poloidal=0, element=pl.main_ion))
        nimp0.append(pl.ion_dens.sel(rho_poloidal=0, element=elem))
        te0.append(pl.el_temp.sel(rho_poloidal=0))
        ti0.append(pl.ion_temp.sel(element=elem, rho_poloidal=0))

    el_dens = xr.concat(el_dens, "run_name").assign_coords({"run_name": runs})
    stdev = el_dens.std("run_name")
    pl.el_dens = el_dens.mean("run_name")
    pl.el_dens_hi = pl.el_dens + stdev
    pl.el_dens_lo = pl.el_dens - stdev

    ion_dens = xr.concat(ion_dens, "run_name").assign_coords({"run_name": runs})
    stdev = ion_dens.std("run_name")
    pl.ion_dens = ion_dens.mean("run_name")
    pl.ion_dens_hi = pl.ion_dens + stdev
    pl.ion_dens_lo = pl.ion_dens - stdev

    neutral_dens = xr.concat(neutral_dens, "run_name").assign_coords({"run_name": runs})
    stdev = neutral_dens.std("run_name")
    pl.neutral_dens = neutral_dens.mean("run_name")
    pl.neutral_dens_hi = pl.neutral_dens + stdev
    pl.neutral_dens_lo = pl.neutral_dens - stdev

    el_temp = xr.concat(el_temp, "run_name").assign_coords({"run_name": runs})
    stdev = el_temp.std("run_name")
    pl.el_temp = el_temp.mean("run_name")
    pl.el_temp_hi = pl.el_temp + stdev
    pl.el_temp_lo = pl.el_temp - stdev

    ion_temp = xr.concat(ion_temp, "run_name").assign_coords({"run_name": runs})
    stdev = ion_temp.std("run_name")
    pl.ion_temp = ion_temp.mean("run_name")
    pl.ion_temp_hi = pl.ion_temp + stdev
    pl.ion_temp_lo = pl.ion_temp - stdev

    meanz = xr.concat(meanz, "run_name").assign_coords({"run_name": runs})
    stdev = meanz.std("run_name")
    pl.meanz = meanz.mean("run_name")
    pl.meanz_hi = pl.meanz + stdev
    pl.meanz_lo = pl.meanz - stdev

    zeff = xr.concat(zeff, "run_name").assign_coords({"run_name": runs})
    stdev = zeff.std("run_name")
    pl.zeff = zeff.mean("run_name")
    pl.zeff_hi = pl.zeff + stdev
    pl.zeff_lo = pl.zeff - stdev

    pressure_th = xr.concat(pressure_th, "run_name").assign_coords({"run_name": runs})
    stdev = pressure_th.std("run_name")
    pl.pressure_th = pressure_th.mean("run_name")
    pl.pressure_th_hi = pl.pressure_th + stdev
    pl.pressure_th_lo = pl.pressure_th - stdev

    run_name = f"RUN{run}{run_add}"
    runs.append(run_name)
    descr = f"Average over runs {runs[0]}-{runs[-1]}"
    run_dict[run_name] = descr
    pl_dict[run_name] = deepcopy(pl)
    bckc_dict[run_name] = deepcopy(bckc)

    if write:
        hda_tree.write(
            pl,
            pulse_to_write,
            "HDA",
            data=data,
            bckc=bckc,
            descr=descr,
            run_name=run_name,
            verbose=True,
            force=force,
        )

    if save_pickle or write:
        save_to_pickle(
            pl, raw_data, data, bckc, pulse=pl.pulse, name=run_name, force=force
        )

    # cols = cm.rainbow(np.linspace(0, 1, len(pl_dict.keys())+2))
    # colors = {}
    # for i, run_name in enumerate(runs):
    #     colors[run_name] = cols[i]
    #
    # plt.figure()
    # for run_name, pl in pl_dict.items():
    #     pl.el_temp.sel(t=t).plot(color=colors[run_name])
    #
    # plt.figure()
    # for run_name, pl in pl_dict.items():
    #     pl.ion_temp.sel(t=t, element=elem).plot(color=colors[run_name])
    #
    # plt.figure()
    # for run_name, pl in pl_dict.items():
    #     pl.el_dens.sel(t=t).plot(color=colors[run_name])
    #
    # plt.figure()
    # for run_name, pl in pl_dict.items():
    #     pl.ion_dens.sel(t=t, element=elem).plot(color=colors[run_name])
    #
    # plt.figure()
    # for run_name, pl in pl_dict.items():
    #     pl.el_temp.sel(rho_poloidal=0).plot(color=colors[run_name])
    #     pl.ion_temp.sel(rho_poloidal=0, element=elem).plot(
    #         color=colors[run_name], linestyle="dashed"
    #     )
    # plt.ylim(0,)
    # plt.title("Te(0) and Ti(0)")
    # plt.ylabel("(eV)")
    #
    # plt.figure()
    # for run_name, pl in pl_dict.items():
    #     pl.el_dens.sel(rho_poloidal=0).plot(color=colors[run_name])
    # plt.ylim(0,)
    # plt.title("Ne(0)")
    # plt.ylabel("(m$^{-3}$)")

    return pl_dict, raw_data, data, bckc_dict, run_dict


def write_profile_scans(
    pl_dict, raw_data, data, bckc_dict, run_dict, modelling=True, force=False
):
    for run_name, descr in run_dict.items():
        if modelling:
            pulse_to_write = pl_dict[run_name].pulse + 25000000
        else:
            pulse_to_write = pl_dict[run_name].pulse

        hda_tree.write(
            pl_dict[run_name],
            pulse_to_write,
            "HDA",
            data=data,
            bckc=bckc_dict[run_name],
            descr=descr,
            run_name=run_name,
            force=force,
        )

        save_to_pickle(
            pl_dict[run_name],
            raw_data,
            data,
            bckc_dict[run_name],
            pulse=pl_dict[run_name].pulse,
            name=run_name,
        )


def read_profile_scans(pulse, plotfig=False, savefig=False, run_add=""):
    runs = np.arange(60, 76 + 1)
    pl_dict = {}
    bckc_dict = {}
    for run in runs:
        run_name = f"RUN{run}{run_add}"
        pl, raw_data, data, bckc = load_pickle(pulse, run_name)
        pl_dict[run_name] = deepcopy(pl)
        bckc_dict[run_name] = deepcopy(bckc)

        if plotfig or savefig:
            plot_results(pl, raw_data, data, bckc, savefig=savefig, name=run_name)
            if not savefig:
                input("Press any key to continue")

    return pl_dict, raw_data, data, bckc_dict


def plot_profile_parametrisation(pl_dict, tgood=0.08, savefig=True):
    plt.figure()
    (1.0e-19 * pl_dict["RUN61"].el_dens.sel(t=tgood, method="nearest")).plot(
        label="broad"
    )
    (1.0e-19 * pl_dict["RUN69"].el_dens.sel(t=tgood, method="nearest")).plot(
        label="peaked"
    )
    plt.title("Electron density")
    plt.ylabel("(m$^{-19}$)")
    plt.legend()
    if savefig:
        plots.save_figure(fig_name="HDA_Ne_profiles")

    plt.figure()
    (1.0e-3 * pl_dict["RUN69"].el_temp.sel(t=tgood, method="nearest")).plot(
        label="broad"
    )
    (1.0e-3 * pl_dict["RUN74"].el_temp.sel(t=tgood, method="nearest")).plot(
        label="peaked"
    )
    plt.title("Electron temperature")
    plt.ylabel("(keV)")
    plt.legend()
    if savefig:
        plots.save_figure(fig_name="HDA_Te_profiles")

    plt.figure()
    (
        1.0e-3
        * pl_dict["RUN74"].ion_temp.sel(element="h").sel(t=tgood, method="nearest")
    ).plot(label="broad")
    (
        1.0e-3
        * pl_dict["RUN68"].ion_temp.sel(element="h").sel(t=tgood, method="nearest")
    ).plot(label="peaked")
    (1.0e-3 * pl_dict["RUN74"].el_temp.sel(t=tgood, method="nearest")).plot(
        label="Te reference", linestyle="dashed", color="black"
    )
    plt.title("Ion temperature")
    plt.ylabel("(keV)")
    plt.legend()
    if savefig:
        plots.save_figure(fig_name="HDA_Ti_profiles")

    plt.figure()
    (
        1.0e-19
        * pl_dict["RUN61"].ion_dens.sel(element="ar").sel(t=tgood, method="nearest")
    ).plot(label="flat")
    (
        1.0e-19
        * pl_dict["RUN62"].ion_dens.sel(element="ar").sel(t=tgood, method="nearest")
    ).plot(label="peaked")
    plt.title("Ar density")
    plt.ylabel("(m$^{-19}$)")
    plt.legend()
    if savefig:
        plots.save_figure(fig_name="HDA_NAr_profiles")


def find_best_profiles(
    pulse=None,
    pl_dict=None,
    raw_data=None,
    data=None,
    bckc_dict=None,
    astra_dict=None,
    tgood=0.08,
    perc_err=0.2,
    savefig=False,
    sxr=True,
    cxrs=True,
    minmax=False,
    tmax=4.01,
    astra_rev="",
    exclude=["63", "65", "67"],
):
    """
    pl_dict, raw_data, data, bckc_dict, astra_dict = tests.find_best_profiles(pulse=9783)
    pl, bckc = tests.find_best_profiles(pl_dict=pl_dict, raw_data=raw_data, data=data, bckc_dict=bckc_dict, astra_dict=astra_dict, savefig=savefig, tgood=0.08, perc_err=0.2, tmax=4.5, minmax=True)
    """

    def average_runs(pl_dict, bckc=None, good_dict=None, tgood=0.08, minmax=False):
        """
        Average results from different runs
        """

        pl_attrs = [
            "el_temp",
            "ion_temp",
            "el_dens",
            "ion_dens",
            "fast_dens",
            "zeff",
            "vloop",
            "wp",
            "wth",
        ]

        pl_to_avrg = {}
        for key in pl_attrs:
            pl_to_avrg[key] = []

        runs = []
        for run_name in pl_dict.keys():
            good_run = True
            if good_dict is not None:
                good_run = good_dict[run_name].sel(t=tgood, method="nearest")
            if good_run:

                runs.append(run_name)
                for key in pl_attrs:
                    pl_to_avrg[key].append(getattr(pl_dict[run_name], key))

        pl = deepcopy(pl_dict[run_name])
        for key in pl_attrs:
            data = xr.concat(pl_to_avrg[key], "run_name").assign_coords(
                {"run_name": runs}
            )
            avrg = data.mean("run_name")
            if minmax:
                setattr(pl, f"{key}_hi", data.max("run_name"))
                setattr(pl, f"{key}_lo", data.min("run_name"))
            else:
                stdev = data.std("run_name")
                setattr(pl, f"{key}_hi", avrg + stdev)
                setattr(pl, f"{key}_lo", avrg - stdev)
            setattr(pl, key, avrg)

        pl.avrg_runs = runs

        return pl

    def compare_runs(
        astra_dict,
        val,
        perc_err=0.2,
        key="",
        good_dict=None,
        chi2_dict=None,
        max_val=False,
    ):
        """
        Find astra runs with parameters < percentage error of the experimental value
        """
        if good_dict is None:
            good_dict = {}
        if chi2_dict is None:
            chi2_dict = {}

        for run_name in astra_dict.keys():
            val_astra = astra_dict[run_name][key].interp(t=val.t)
            if len(val_astra.shape) == 2:
                val_astra = val_astra.sel(rho_poloidal=0, method="nearest")

            if max_val:
                good_tmp = val_astra < val
                perc_err = 0.01
            else:
                good_tmp = np.abs(val_astra - val) / val < perc_err

            if run_name not in good_dict:
                good_dict[run_name] = deepcopy(val_astra)
                good_dict[run_name].values = np.array([True] * len(val_astra))

            good_dict[run_name] *= good_tmp

            if run_name not in chi2_dict:
                chi2_dict[run_name] = np.full_like(val_astra, 0.0)
            chi2_dict[run_name] += np.abs((val_astra - val) / (val * perc_err)) ** 2

        return good_dict, chi2_dict

    def plot_compare(
        astra_dict=None,
        pl_dict=None,
        bckc_dict=None,
        raw=None,
        data=None,
        perc_err=0.0,
        instrument="",
        quantity="",
        title="",
        xlabel=None,
        ylabel=None,
        label=None,
        const=1.0,
        all_runs=[],
        good_runs=[],
        tgood=None,
        profile=False,
        normalize=False,
        figure=True,
        ylim=(None, None),
        savefig=False,
    ):

        ylim_data = None
        rho_min = None

        if figure:
            plt.figure()

        if raw is not None:
            value = raw * const
            if len(value.shape) > 1:
                if profile:
                    value = value.sel(t=tgood, method="nearest")
                else:
                    value = value.sel(rho_poloidal=0, method="nearest")

            if normalize:
                norm = np.nanmax(value.values)
                value /= norm

            value.plot(color="black", alpha=0.5)

        if data is not None:
            if hasattr(data, "attrs"):
                if "rho_min" in data.attrs:
                    dim = "diode_arrays_filter_4_coords"
                    rho_min = data.rho_min.sel(t=tgood, method="nearest")

            value = data * const
            if "error" in data.attrs:
                err = (data.attrs["error"] * const) ** 2
            else:
                err = xr.full_like(value, 0.0)

            if len(value.shape) > 1:
                if profile:
                    value = value.sel(t=tgood, method="nearest")
                    err = err.sel(t=tgood, method="nearest")
                else:
                    value = value.sel(rho_poloidal=0, method="nearest")
                    err = err.sel(rho_poloidal=0, method="nearest")

            err += (value * perc_err) ** 2
            err = np.sqrt(err)

            if normalize:
                norm = np.nanmax(value)
                value /= norm
                err /= norm

            if rho_min is not None:
                value = value.assign_coords(rho_min=(dim, rho_min))
                value = value.swap_dims({dim: "rho_min"})
                err = err.assign_coords(rho_min=(dim, rho_min))
                err = err.swap_dims({dim: "rho_min"})

            value.plot(
                linewidth=3,
                color="black",
                marker="o",
                alpha=0.5,
                label=label,
            )
            plt.fill_between(
                value.coords[value.dims[0]],
                (value - err),
                (value + err),
                color="black",
                alpha=0.5,
            )
            ylim_data = [np.mean(value - err), np.mean(value + err)]

        ylim_lo = []
        ylim_hi = []
        for run_name in all_runs:
            if astra_dict is not None:
                value = astra_dict[run_name][quantity] * const
                label = "ASTRA"
            elif pl_dict is not None:
                value = pl_dict[run_name][quantity] * const
                label = "HDA"
            elif bckc_dict is not None:
                value = bckc_dict[run_name][instrument][quantity] * const
                label = "bckc"
            else:
                print("Input either of the following: astra_dict, pl_dict, bckc_dict")
                raise ValueError

            if len(value.shape) == 2:
                if profile:
                    value = value.sel(t=tgood, method="nearest")
                else:
                    value = value.sel(rho_poloidal=0, method="nearest")

            if normalize:
                norm = np.nanmax(value)
                value /= norm

            if rho_min is not None:
                value = value.assign_coords(rho_min=(dim, rho_min))
                value = value.swap_dims({dim: "rho_min"})

            value.plot(alpha=0.5, color="red", linestyle="dashed")
            if run_name in good_runs:
                ylim_lo.append(value.min())
                ylim_hi.append(value.max())
                value.plot(label=f"{label} {run_name}", linewidth=4, alpha=0.8)

        _ylim = [np.min(ylim_lo), np.max(ylim_hi)]
        if ylim_data is not None:
            _ylim = [np.min([_ylim, ylim_data]), np.max([_ylim, ylim_data])]

        if ylim[0] is not None:
            _ylim[0] = ylim[0]
        if ylim[1] is not None:
            _ylim[1] = ylim[1]
        if _ylim[0] != 0:
            _ylim[0] *= 0.9
        _ylim[1] *= 1.1

        plt.ylim(_ylim)
        ylim = plt.ylim()
        if good_dict is not None and tgood is not None and not profile:
            plt.vlines(tgood, ylim[0], ylim[1], linestyle="dashed", color="black")

        name = f"ASTRA_compare_{all_runs[0]}-{all_runs[-1]}"
        _title = f"{pulse} {title}"
        if profile:
            _title += f" t={int(tgood*1.e3)} ms"
            name += "_profile"
        else:
            name += "_time_evol"
        if tgood is not None:
            name += f"_{int(tgood * 1.e3)}_ms"
        if len(instrument) > 0:
            name += f"_{instrument}"
        if "/" in quantity:
            name += f"_{quantity.replace('/', '_')}"
        else:
            name += f"_{quantity}"

        plt.title(_title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.legend(fontsize=10)

        if savefig:
            figname = plots.get_figname(pulse=pulse, name=name)
            plots.save_figure(fig_name=figname)

    def add_missing_cxrs(pl, raw_data, data):
        reader_st40 = ST40Reader(pl.pulse, pl.tstart, pl.tend, tree="ST40")
        instrument = "princeton"
        quantity = "ti"

        R_orig, _ = reader_st40._get_data(
            "spectrom", "princeton.cxsfit", ".input:r_orig", 0
        )
        phi_orig, _ = reader_st40._get_data(
            "spectrom", "princeton.cxsfit", ".input:phi_orig", 0
        )
        x_orig = R_orig * np.cos(phi_orig)
        y_orig = R_orig * np.sin(phi_orig)
        z_orig, dims = reader_st40._get_data(
            "spectrom", "princeton.cxsfit", ".input:z_orig", 0
        )

        R_pos, _ = reader_st40._get_data(
            "spectrom", "princeton.cxsfit", ".input:r_pos", 0
        )
        phi_pos, _ = reader_st40._get_data(
            "spectrom", "princeton.cxsfit", ".input:phi_pos", 0
        )
        x_pos = R_pos * np.cos(phi_pos)
        y_pos = R_pos * np.sin(phi_pos)
        z_pos, dims = reader_st40._get_data(
            "spectrom", "princeton.cxsfit", ".input:z_pos", 0
        )

        location = np.array([x_orig, y_orig, z_orig]).transpose()
        direction = np.array([x_pos, y_pos, z_pos]).transpose() - location

        from indica.converters.lines_of_sight_jw import LinesOfSightTransform

        rev = "3"
        values, dims = reader_st40._get_data(
            "spectrom", "princeton.cxsfit_out", ":ti", rev
        )
        # Loop on channels and times to non-nil values
        ch_ind = []
        t_ind = []
        for i in range(values.shape[1]):
            if any(values[:, i]):
                ch_ind.append(i)
                t_ind.append(np.where(values[:, i] > 0)[0])
        t_ind = np.arange(np.min(t_ind), np.max(t_ind) + 1)

        err, dims = reader_st40._get_data(
            "spectrom", "princeton.cxsfit_out", ":ti_err", rev
        )
        times = dims[1][t_ind]
        location = location[ch_ind, :]
        direction = direction[ch_ind, :]
        R_nbi = dims[0][ch_ind]
        x_nbi = x_pos[ch_ind]
        # y_nbi = y_pos[ch_ind]
        # z_nbi = z_pos[ch_ind]
        values = values[t_ind, :]
        values = values[:, ch_ind]
        err = err[t_ind, :]
        err = err[:, ch_ind]

        # restrict to channels with data only
        transform = []
        dl_nbi = 0.2
        for i in range(len(R_nbi)):
            trans = LinesOfSightTransform(
                location[i, :],
                direction[i, :],
                f"{instrument}_{quantity}",
                reader_st40.MACHINE_DIMS,
            )
            x, y = trans.convert_to_xy(0, trans.x2, 0)
            R, z = trans.convert_to_Rz(0, trans.x2, 0)

            trans.x, trans.y, trans.z, trans.R = x, y, z, R

            trans.R_nbi = R_nbi[i]

            rho_equil, _ = pl.flux_coords.convert_from_Rz(trans.R, trans.z, t=times)
            rho = rho_equil.interp(t=times, method="linear")
            rho = xr.where(rho >= 0, rho, 0.0)
            rho.coords[trans.x2_name] = trans.x2
            trans.rho = rho

            dn_nbi = int(dl_nbi / trans.dl / 2.0)
            ind = np.argmin(np.abs(x.values - x_nbi[i]))
            nbi_x2 = trans.x2[ind]
            in_x2 = trans.x2[ind - dn_nbi]
            out_x2 = trans.x2[ind + dn_nbi]

            trans.rho_nbi = rho.sel(princeton_ti_los_position=nbi_x2)

            rho_tmp = rho.sel(princeton_ti_los_position=slice(in_x2, out_x2))
            trans.rho_in = rho_tmp.min("princeton_ti_los_position")
            trans.rho_out = rho_tmp.max("princeton_ti_los_position")

            transform.append(trans)

        coords = [
            ("t", times),
            (transform[0].x1_name, ch_ind),
        ]
        error = DataArray(err, coords).sel(
            t=slice(reader_st40._tstart, reader_st40._tend)
        )
        meta = {
            "datatype": "ti",
            "error": error,
            "transform": transform,
        }
        quant_data = DataArray(
            values,
            coords,
            attrs=meta,
        ).sel(t=slice(reader_st40._tstart, reader_st40._tend))

        quant_data.name = "princeton" + "_" + "ti"
        quant_data.attrs["revision"] = rev

        data["princeton"] = {}
        data["princeton"]["cxsfit_bgnd"] = quant_data

        # Multi-gaussian fit
        values, dims = reader_st40._get_data(
            "spectrom", "princeton.cxsfit_out", ":ti", 5
        )
        err, dims = reader_st40._get_data(
            "spectrom", "princeton.cxsfit_out", ":ti_err", 5
        )

        values = values[t_ind, :]
        values = values[:, ch_ind]
        err = err[t_ind, :]
        err = err[:, ch_ind]

        error = DataArray(err, coords).sel(
            t=slice(reader_st40._tstart, reader_st40._tend)
        )
        meta = {
            "datatype": "ti",
            "error": error,
            "transform": transform,
        }
        quant_data = DataArray(
            values,
            coords,
            attrs=meta,
        ).sel(t=slice(reader_st40._tstart, reader_st40._tend))

        quant_data.name = "princeton" + "_" + "ti"
        quant_data.attrs["revision"] = rev

        data["princeton"]["cxsfit_full"] = quant_data

        return raw_data, data

    def add_missing_sxr(pl, raw_data, data, rotate=True):
        reader_st40 = ST40Reader(pl.pulse, pl.tstart, pl.tend, tree="ST40")

        plt.close("all")
        plt.ioff()
        revision = 0
        sxr = reader_st40.get(
            "sxr", "diode_arrays", revision=revision, quantities=["filter_4"]
        )
        plt.ion()

        raw_data["sxr"] = sxr
        for kquant in sxr.keys():
            binned_data = bin_in_time_dt(pl.tstart, pl.tend, pl.dt, sxr[kquant])
            # rotate channels to test geometry errors
            if rotate:
                for t in binned_data.t:
                    tmp = binned_data.sel(t=t).values
                    binned_data.loc[dict(t=t)] = np.flip(tmp)

            binned_data.attrs["transform"].set_equilibrium(pl.equilibrium, force=True)
            geom_attrs = remap_diagnostic(binned_data, pl.flux_coords)

            # Temp fix for LOS going beyond remit (Jon's version should fix this)
            rho = geom_attrs["rho"]
            for t in rho.t:
                rho.loc[dict(t=t)] = xr.where(
                    geom_attrs["rho"].R < 0.9, geom_attrs["rho"].sel(t=t), np.nan
                )
            rho_min = rho.min("diode_arrays_filter_4_los_position")

            for t in rho_min.t:
                zmag = pl.equilibrium.zmag.sel(t=t, method="nearest")
                Rmag = pl.equilibrium.rmag.sel(t=t, method="nearest")
                impact = xr.where(
                    rho.sel(t=t).R >= Rmag,
                    np.sqrt(
                        (rho.sel(t=t).R - Rmag) ** 2 + (rho.sel(t=t).z - zmag) ** 2
                    ),
                    np.nan,
                )
                impact_los_pos = impact.diode_arrays_filter_4_los_position[
                    impact.argmin("diode_arrays_filter_4_los_position")
                ]
                zimpact = rho.z.interp(
                    diode_arrays_filter_4_los_position=impact_los_pos
                )

                rho_min.loc[dict(t=t)] = xr.where(
                    zimpact < 0,
                    -rho_min.sel(t=t),
                    rho_min.sel(t=t),
                )

            geom_attrs["rho"] = rho
            geom_attrs["rho_min"] = rho_min
            for kattrs in geom_attrs:
                binned_data.attrs[kattrs] = geom_attrs[kattrs]
        data["sxr"] = {"filter_4": binned_data}

        return raw_data, data

    def add_missing_sxr_atomdat(pl_dict):
        pl = pl_dict[list(pl_dict.keys())[0]]

        # SXR cooling functions
        adasreader = ADASReader()
        pls_prs = {"pls": "15", "prs": "15"}
        adf11, pls, prs, power_loss_sxr, lz_sxr = {}, {}, {}, {}, {}
        for elem in pl_avrg.elements:
            adf11 = pls_prs
            pls = adasreader.get_adf11("pls", elem, adf11["pls"])
            prs = adasreader.get_adf11("prs", elem, adf11["prs"])
            power_loss_sxr[elem] = PowerLoss(pls, prs)

        # Takes too much time to go through the whole process,
        # reduce to case of no neutral density...
        # TODO: fix this before it's too late !!!
        Te = pl.el_temp.sel(t=pl.t[0]).drop("t")
        Te.values = np.linspace(10, 10.0e3, len(Te)).transpose()
        Ne = xr.full_like(Te, 5.0e19)
        lz = {}
        for elem in pl.elements:
            fz_tmp = pl.fract_abu[elem](Ne, Te)
            lz[elem] = power_loss_sxr[elem](Ne, Te, fz_tmp).transpose().values
            lz[elem] = DataArray(
                lz[elem],
                coords=[
                    ("electron_temperature", Te),
                    ("ion_charges", np.arange(0, lz[elem].shape[1])),
                ],
            )

        # Add missing SXR attributes to Plasma class
        for run in pl_dict.keys():
            pl_dict[run].power_loss_sxr = deepcopy(power_loss_sxr)
            pl_dict[run].prad_sxr = deepcopy(xr.full_like(pl.prad, np.nan))
            sxr_rad = xr.zeros_like(pl_dict[run].tot_rad)
            sxr_rad.name = "sxr_radiation"
            sxr_rad.attrs["datatype"] = ("radiation", "sxr")
            pl_dict[run].sxr_rad = sxr_rad

            pl_dict[run].lz_sxr = deepcopy(pl.lz)
            for elem in lz.keys():
                pl_dict[run].lz_sxr[elem] = xr.full_like(pl.lz[elem], np.nan)
                lz_interp = (
                    lz[elem]
                    .interp(electron_temperature=pl_dict[run].el_temp)
                    .drop("electron_temperature")
                )
                pl_dict[run].lz_sxr[elem].values = lz_interp.values

    def calc_sxr_los(pl, data, bckc):
        for elem in pl.elements:
            pl.sxr_rad.loc[dict(element=elem)] = (
                pl.lz_sxr[elem].sum("ion_charges")
                * pl.el_dens
                * pl.ion_dens.sel(element=elem)
            )

        # Interpolate on diagnostic LOSs and back-calculate LOS integral
        initialize_bckc("sxr", "filter_4", data, bckc=bckc)
        bckc_tmp = bckc["sxr"]["filter_4"]

        sxr_rad_interp = pl.sxr_rad.sum("element").interp(rho_poloidal=bckc_tmp.rho)
        sxr_rad_interp = xr.where(
            (bckc_tmp.rho <= 1) * np.isfinite(sxr_rad_interp),
            sxr_rad_interp,
            0,
        )
        x2_name = "diode_arrays_filter_4_los_position"
        bckc_tmp = sxr_rad_interp.sum(x2_name) * bckc_tmp.dl
        bckc["sxr"]["filter_4"].values = bckc_tmp.values
        bckc_dict[run] = deepcopy(bckc)

    """
    Of a given set of ASTRA runs based on HDA profiles, select the ones that better match
    other measurements and parameters (e.g. EFIT stored energy, ...)
    """

    if pl_dict is None:
        # Read scans from pickle files
        pl_dict, raw_data, data, bckc_dict = read_profile_scans(pulse)
        runs = list(pl_dict)
        if len(exclude) > 0:
            for exc in exclude:
                kpop = f"RUN{exc}"
                if kpop in pl_dict.keys():
                    pl_dict.pop(kpop)
                if kpop in bckc_dict.keys():
                    bckc_dict.pop(kpop)

        pl_avrg = pl_dict["RUN60"]
        pulse = 13100000 + pl_avrg.pulse
        tstart = 0
        tend = 0.2

        # Changed some attributes...
        for run, pl in pl_dict.items():
            if not hasattr(pl, "power_loss_tot"):
                pl_dict[run].power_loss_tot = pl_dict[run].power_loss
                pl_dict[run].lz_tot = pl_dict[run].lz

        # Read astra
        astra_dict = {}
        reader_astra = ST40Reader(pulse, tstart, tend, tree="ASTRA")
        for run in pl_dict.keys():
            # Read ASTRA results
            revision = run[3:] + astra_rev
            astra_dict[run] = reader_astra.get("", "astra", revision)

            # find q=1 surface
            q_prof = astra_dict[run]["q"]
            rho = q_prof.rho_poloidal
            ind = np.abs(1 - q_prof).argmin("rho_poloidal")
            q1 = deepcopy(ind)
            q1.name = "q1"
            q1 = [rho[i] for i in ind]
            q1 = xr.concat(q1, "time").drop("rho_poloidal")
            q1.name = "q1_astra"
            astra_dict[run]["q1"] = q1

            # Calculate the thermal and EFIT stored energies (ASTRA wth is in reality the total Wth + Wpar + Wperp)
            Pe = (
                astra_dict[run]["ne"]
                * astra_dict[run]["te"]
                * constants.e
                * 1.0e3
                * 1.0e19
            )
            Pi = (
                astra_dict[run]["ne"]
                * astra_dict[run]["ti"]
                * constants.e
                * 1.0e3
                * 1.0e19
            )
            Pth = Pe + Pi
            Pblon = astra_dict[run]["pblon"]
            Pbperp = astra_dict[run]["pbper"]

            volume = astra_dict[run]["volume"]

            # TODO: calculate also Wdia to compare with experiment
            wtot = deepcopy(astra_dict[run]["wth"])
            wtot.name = "astra_wtot"
            wtot.attrs["datatype"] = ("stored_energy", "total")
            weq = deepcopy(astra_dict[run]["wth"])
            weq.name = "astra_weq"
            weq.attrs["datatype"] = ("stored_energy", "equilibrium")
            wth_th = xr.zeros_like(astra_dict[run]["wth"])
            wth_th.name = "astra_wth_th"
            for t in weq.t:
                vol_tmp = volume.sel(t=t, method="nearest")

                wth_th.loc[dict(t=t)] = (
                    3 / 2 * np.trapz(Pth.sel(t=t, method="nearest"), vol_tmp)
                )
                weq.loc[dict(t=t)] = wth_th.sel(t=t) + np.trapz(
                    3 / 4.0 * (Pbperp + Pblon).sel(t=t), vol_tmp
                )
                wtot.loc[dict(t=t)] = weq.sel(t=t) - np.trapz(
                    1 / 4 * (Pblon - Pbperp).sel(t=t), vol_tmp
                )
            astra_dict[run]["wth_th"] = wth_th
            astra_dict[run]["wtot"] = wtot
            astra_dict[run]["weq"] = weq

            # Test increased equilibrium
            # plt.figure()
            # weq_increased = deepcopy(weq)
            # volume_increased = deepcopy(volume)
            # for t in weq.t:
            #     volume_increased.loc[dict(t=t)] = (
            #         volume.sel(t=t, method="nearest").rho_poloidal ** 2.5
            #         * volume.sel(t=t, method="nearest").max()
            #     )
            #     vol_tmp = volume_increased.sel(t=t, method="nearest")
            #
            #     wth_th_tmp = 3 / 2 * np.trapz(Pth.sel(t=t, method="nearest"), vol_tmp)
            #     weq_increased.loc[dict(t=t)] = wth_th_tmp + np.trapz(
            #         3 / 4.0 * (Pbperp + Pblon).sel(t=t), vol_tmp
            #     )
            #
            # if run == "RUN72":
            #     plt.figure()
            #     weq.plot(label="ASTRA volume")
            #     weq_increased.plot(label="Increased volume")
            #     plt.legend()
            #
            #     plt.figure()
            #     volume.sel(t=0.08, method="nearest").plot(label="ASTRA")
            #     volume_increased.sel(t=0.08, method="nearest").plot(label="ASTRA")
            #     plt.legend()

            # Substitute HDA wp/vloop and other quantities with values from ASTRA
            pl_dict[run].wp.values = (
                astra_dict[run]["wth"].interp(t=pl_dict[run].t).values
            )
            pl_dict[run].vloop.values = (
                astra_dict[run]["upl"].interp(t=pl_dict[run].t).values
            )
            pl_dict[run].ion_dens.loc[dict(element=pl_dict[run].main_ion)] = (
                (astra_dict[run]["ni"] * 1.0e19)
                .interp(t=pl_dict[run].t)
                .interp(rho_poloidal=pl_dict[run].rho)
            )
            pl_dict[run].fast_dens.values = (
                (astra_dict[run]["nf"] * 1.0e19)
                .interp(t=pl_dict[run].t)
                .interp(rho_poloidal=pl_dict[run].rho)
            )

        # Read SXR data if it isn't already in the data structure
        if sxr and ("sxr" not in raw_data):
            raw_data, data = add_missing_sxr(pl, raw_data, data)

        if cxrs and ("princeton" not in data):
            raw_data, data = add_missing_cxrs(pl, raw_data, data)

        if not hasattr(pl_avrg, "lz_sxr"):
            add_missing_sxr_atomdat(pl_dict)

            # Back-calculated SXR signals
            for run in pl_dict.keys():
                bckc = bckc_dict[run]
                calc_sxr_los(pl_dict[run], data, bckc)
                bckc_dict[run] = bckc

        return pl_dict, raw_data, data, bckc_dict, astra_dict

    pulse = pl_dict["RUN60"].pulse

    # Selection of good runs
    # Stored energy inside uncertainty band of EFIT
    good_dict, chi2_dict = compare_runs(
        astra_dict, data["efit"]["wp"], perc_err=perc_err, key="wth"
    )

    # Central electron temperature < current atomic data limit of 4 keV
    # val = xr.full_like(data["efit"]["wp"], tmax)
    # good_dict, _ = compare_runs(
    #     astra_dict,
    #     val,
    #     key="te",
    #     good_dict=good_dict,
    #     max_val=True,
    # )

    # Central electron temperature < current atomic data limit of 4 keV
    # val = xr.full_like(data["efit"]["wp"], 20)
    # good_dict, _ = compare_runs(
    #     astra_dict,
    #     val,
    #     key="ti",
    #     good_dict=good_dict,
    #     max_val=True,
    # )

    all_runs = list(pl_dict)

    good_runs = []
    chi2 = []
    for run_name in good_dict.keys():
        if good_dict[run_name].sel(t=tgood, method="nearest") == True:
            good_runs.append(run_name)
            chi2.append(chi2_dict[run_name].sel(t=tgood, method="nearest").values)
    best_run = good_runs[np.argmin(chi2)]

    # TODO: add comparisons to all experimental quantities
    const = 1.0e-3
    plot_compare(
        astra_dict=astra_dict,
        raw=raw_data["efit"]["wp"],
        data=data["efit"]["wp"],
        quantity="wth",
        title="Stored energy",
        xlabel="Time (s)",
        ylabel="(kJ)",
        label="EFIT",
        const=const,
        perc_err=perc_err,
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        ylim=(0, None),
        savefig=savefig,
    )

    plot_compare(
        astra_dict=astra_dict,
        raw=raw_data["vloop"],
        data=data["vloop"],
        quantity="upl",
        title="Loop voltage",
        xlabel="Time (s)",
        ylabel="(V)",
        label="MAG",
        perc_err=perc_err,
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        ylim=(0, None),
        savefig=savefig,
    )

    plot_compare(
        astra_dict=astra_dict,
        quantity="q1",
        title="q=1 surface position",
        xlabel="Time (s)",
        ylabel="(rho_poloidal)",
        label="ASTRA",
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        savefig=savefig,
    )

    plot_compare(
        astra_dict=astra_dict,
        quantity="ne",
        title="Central electron density",
        xlabel="Time (s)",
        ylabel="(10$^{19}$ m$^{-3}$)",
        label="ASTRA",
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        ylim=(0, None),
        savefig=savefig,
    )

    plot_compare(
        astra_dict,
        quantity="te",
        title="Central electron temperature",
        xlabel="Time (s)",
        ylabel="(keV)",
        label="ASTRA",
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        ylim=(0, None),
        savefig=savefig,
    )

    plot_compare(
        astra_dict=astra_dict,
        quantity="ti",
        title="Central ion temperature",
        xlabel="Time (s)",
        ylabel="(keV)",
        label="ASTRA",
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        ylim=(0, None),
        savefig=savefig,
    )

    plot_compare(
        astra_dict=astra_dict,
        quantity="ne",
        title="Electron density",
        ylabel="(10$^{19}$ m$^{-3}$)",
        label="ASTRA",
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        profile=True,
        ylim=(0, None),
        savefig=savefig,
    )

    plot_compare(
        astra_dict=astra_dict,
        quantity="te",
        title="Electron temperature",
        ylabel="(keV)",
        label="ASTRA",
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        profile=True,
        ylim=(0, None),
        savefig=savefig,
    )

    plot_compare(
        astra_dict=astra_dict,
        quantity="ti",
        title="Ion temperature",
        ylabel="(keV)",
        label="ASTRA",
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        profile=True,
        ylim=(0, None),
        savefig=savefig,
    )

    instrument = "xrcs"
    quantity = "int_n3/int_tot"
    plot_compare(
        bckc_dict=bckc_dict,
        raw=raw_data[instrument][quantity],
        data=data[instrument][quantity],
        instrument=instrument,
        quantity=quantity,
        title="n3/w line ratio",
        ylabel="",
        label="XRCS n3/w",
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        ylim=(0, None),
        savefig=savefig,
    )

    instrument = "xrcs"
    quantity = "int_w"
    plot_compare(
        bckc_dict=bckc_dict,
        raw=raw_data[instrument][quantity],
        data=data[instrument][quantity],
        instrument=instrument,
        quantity=quantity,
        title="w line intensity",
        ylabel="",
        label="XRCS w",
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        ylim=(0, None),
        savefig=savefig,
    )

    instrument = "lines"
    quantity = "brems"
    plot_compare(
        bckc_dict=bckc_dict,
        raw=raw_data[instrument][quantity],
        data=data[instrument][quantity],
        instrument=instrument,
        quantity=quantity,
        title="Bremsstrahlung",
        ylabel="(a.u.)",
        label="DIODES",
        all_runs=all_runs,
        good_runs=good_runs,
        tgood=tgood,
        ylim=(0, None),
        savefig=savefig,
    )

    if "sxr" in data.keys():
        instrument = "sxr"
        quantity = "filter_4"
        plot_compare(
            bckc_dict=bckc_dict,
            data=data[instrument][quantity],
            instrument=instrument,
            quantity=quantity,
            title="SXR camera",
            ylabel="(a.u.)",
            label="SXR filter 4",
            profile=True,
            perc_err=perc_err,
            all_runs=all_runs,
            good_runs=good_runs,
            tgood=tgood,
            ylim=(0, None),
            savefig=savefig,
            normalize=True,
        )

    pl = average_runs(pl_dict, good_dict=good_dict, tgood=tgood, minmax=minmax)

    # pl = deepcopy(pl_dict[best_run])
    # pl.run = best_run
    bckc = deepcopy(bckc_dict[best_run])

    # data["efit"]["wp"].attrs["error"] = np.abs(data["efit"]["wp"] * perc_err)
    initialize_bckc("efit", "wp", data, bckc=bckc)
    bckc["efit"]["wp"].values = pl.wp.values

    data["mag"] = {}
    data["mag"]["vloop"] = data["vloop"]
    # data["mag"]["vloop"].attrs["error"].values = np.sqrt(
    #     data["mag"]["vloop"].error ** 2 + (data["mag"]["vloop"] * perc_err) ** 2
    # ).values
    initialize_bckc("mag", "vloop", data, bckc=bckc)
    bckc["mag"]["vloop"] = pl.vloop

    name = "ASTRA_compare_average"
    # plots.profiles(
    #     pl,
    #     data=data,
    #     bckc=bckc,
    #     savefig=savefig,
    #     name=best_run,
    # )
    plots.profiles(
        pl,
        data=data,
        bckc=bckc,
        savefig=savefig,
        name=name,
        ploterr=True,
        tplot=tgood,
    )
    plots.time_evol(
        pl,
        data,
        bckc=bckc,
        savefig=savefig,
        name=name,
        ploterr=True,
    )
    # plots.compare_data_bckc(
    #     data,
    #     bckc,
    #     raw_data=raw_data,
    #     pulse=pulse,
    #     savefig=savefig,
    #     name=name,
    #     ploterr=False,
    # )

    # import pandas as pd
    # runs_csv = ["RUN64", "RUN69", "RUN70", "RUN71"]
    # for run in runs_csv:
    #     pulse = pl_dict[run].pulse
    #     rho = pl_dict[run].rho.values
    #     ion_temp = pl_dict[run].ion_temp.sel(t=0.08,method="nearest").sel(element="h").values
    #     el_temp = pl_dict[run].el_temp.sel(t=0.08,method="nearest").values
    #     el_dens = pl_dict[run].el_dens.sel(t=0.08,method="nearest").values
    #     ar_dens = pl_dict[run].ion_dens.sel(t=0.08,method="nearest").sel(element="ar").values
    #     to_write = {
    #         "Rho-poloidal": rho,
    #         "Te profile (eV)": el_temp,
    #         "Ti profile (eV)": ion_temp,
    #         "Ne profile (eV)": el_dens,
    #         "N_Ar profile (eV)": ar_dens,
    #         "Zeff":, zeff
    #     }
    #     df = pd.DataFrame(to_write)
    #     df.to_csv(f"/home/marco.sertoli/data/Indica/{pulse}_{run}_profiles.csv")

    print(f"Best run {best_run}")
    return pl, bckc


#
# def vertical_displacement(
#     pulse=8804,
#     tstart=0.05,
#     tend=0.12,
#     dt=0.01,
#     diagn_ne="nirh1_bin",
#     quant_ne="ne",
#     save_pickle=False,
#     savefig=False,
#     name="vertical_displacement",
#     perc_err=0.05,
#     y1=1.0e19,
#     nsteps=6,
#     ref_pulse=None,
# ):
#     """
#     # These are really good pulses to compare, with constant flat-top-parameters
#     # and goo geometry scan --> flat density profile
#     res = tests.vertical_displacement(
#         pulse=8804,
#         tstart=0.05,
#         tend=0.12,
#         dt=0.01,
#         diagn_ne="nirh1_bin",
#         quant_ne="ne",
#     )
#
#     res = tests.vertical_displacement(
#         pulse=8806,
#         tstart=0.07,
#         tend=0.13,
#         dt=0.01,
#         diagn_ne="nirh1_bin",
#         quant_ne="ne",
#     )
#
#     res = tests.vertical_displacement(
#         pulse=8807,
#         tstart=0.07,
#         tend=0.12,
#         dt=0.015,
#         diagn_ne="nirh1_bin",
#         quant_ne="ne",
#     )
#
#     # The following two are reference and z-shift pulse, but they have different RFX
#     # waveforms and RFX switch-off during the ramp so are not good to compare
#     res = tests.vertical_displacement(
#         pulse=9651,
#         tstart=0.08,
#         tend=0.13,
#         dt=0.01,
#         diagn_ne="nirh1_bin",
#         quant_ne="ne",
#         y1=0.1e19,
#     )
#
#     res = tests.vertical_displacement(
#         pulse=9652,
#         tstart=0.08,
#         tend=0.13,
#         dt=0.01,
#         diagn_ne="nirh1_bin",
#         quant_ne="ne",
#         y1=0.1e19,
#     )
#     """
#
#     def line(x, intercept, slope):
#         return intercept + x * slope
#
#     main_ion = "h"
#     impurities = ("c", "ar", "he")
#
#     raw = ST40data(pulse, tstart - dt * 2, tend + dt * 2)
#     raw.get_all()
#     raw_data = raw.data
#
#     bckc = {}
#     elements = list(main_ion)
#     elements.extend(list(impurities))
#
#     pl = Plasma(tstart=tstart, tend=tend, dt=dt, elements=elements)
#     data = pl.build_data(raw_data, pulse=pulse)
#
#     Ne = deepcopy(pl.Ne_prof)
#     wped_all = np.linspace(12, 4, nsteps)  # [12, 6, 5, 4]
#     peaking_all = np.linspace(1.05, 3, nsteps)  # [1.05, 1.3, 2.0, 4]
#     colors = cm.turbo(np.linspace(0, 1, nsteps))
#
#     Ne_all = []
#     pl_all = []
#     bckc_all = []
#     for wped, peaking in zip(wped_all, peaking_all):
#         Ne.wped = wped
#         Ne.y1 = y1
#         Ne.peaking = peaking
#         Ne.build_profile()
#         Ne_all.append(deepcopy(Ne))
#
#         pl.Ne_prof = deepcopy(Ne)
#
#         bckc = pl.match_interferometer(
#             data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne
#         )
#
#         bckc = pl.interferometer(data, bckc=bckc)
#         bckc = pl.bremsstrahlung(data, bckc=bckc)
#
#         pl_all.append(deepcopy(pl))
#         bckc_all.append(deepcopy(bckc))
#
#     figname = plots.get_figname(pulse=pulse, name=name)
#     cmap = cm.rainbow
#     colors_t = cmap(np.linspace(0, 1, len(pl.time)))
#
#     plt.figure()
#     t = pl.t[int(len(pl.t) / 2)]
#     for pl, c in zip(pl_all, colors):
#         (pl.el_dens.sel(t=t) / 1.0e19).plot(color=c)
#     plt.title(f"{pulse} Density profiles t = {t:.3f}")
#     plt.xlabel("Rho-poloidal")
#     plt.ylabel("(10$^{19}$ m$^{-3}$)")
#     plt.legend()
#     if savefig:
#         plots.save_figure(fig_name=f"{figname}profiles_electron_densities")
#
#     plt.figure()
#     raw = raw_data[diagn_ne][quant_ne]
#     value = data[diagn_ne][quant_ne]
#     error = value.attrs["error"]
#
#     (raw / 1.0e19).plot()
#     plt.plot(value.t, (value) / 1.0e19, "-o", color="red")
#     plt.fill_between(
#         value.t,
#         (value - error) / 1.0e19,
#         (value + error) / 1.0e19,
#         alpha=0.5,
#         color="red",
#     )
#     plt.title(f"{pulse} {diagn_ne.upper()} electron density measurement")
#     plt.ylim(0,)
#     plt.xlabel("Time (s)")
#     plt.ylabel("(10$^{19}$ m$^{-2}$)")
#     plt.legend()
#     if savefig:
#         plots.save_figure(fig_name=f"{figname}data_density_measurement")
#
#     plt.figure()
#     for pl, c in zip(pl_all, colors):
#         x = pl.time
#         y = pl.el_dens.sel(rho_poloidal=0) / 1.0e19
#
#         pfit = np.polyfit(x.values, y.values, 1)
#         slope, intercept = pfit
#
#         yfit = line(x, intercept, slope)
#         yfit.plot(linestyle="dashed", color=c)
#         residuals = y - yfit
#         chi_sq = np.sum((residuals / (y * perc_err)) ** 2)
#         plt.plot(x, y, label=f"slope = {slope:.2f}", color=c)
#         for t, c in zip(pl.time, colors_t):
#             plt.errorbar(
#                 t, y.sel(t=t), perc_err * y.sel(t=t), alpha=0.8, marker="o", color=c
#             )
#
#     plt.title(f"{pulse} Central electron density")
#     plt.xlabel("Time (s)")
#     plt.ylabel("(10$^{19}$ m$^{-3}$)")
#     plt.legend()
#     if savefig:
#         plots.save_figure(fig_name=f"{figname}time_evol_central_densities")
#
#     plt.figure()
#     R = pl.equilibrium.rho.R
#     z = pl.equilibrium.rho.z
#     vmin = np.linspace(1, 0, len(pl.time))
#     for i, t in enumerate(pl.time):
#         rho = pl.equilibrium.rho.sel(t=t, method="nearest")
#         plt.contour(
#             R,
#             z,
#             rho,
#             levels=[1.0],
#             alpha=0.5,
#             cmap=cmap,
#             vmin=vmin[i],
#             vmax=vmin[i] + 1,
#         )
#         plt.plot(
#             pl.equilibrium.rmag.sel(t=t, method="nearest"),
#             pl.equilibrium.zmag.sel(t=t, method="nearest"),
#             color=colors_t[i],
#             marker="o",
#             alpha=0.5,
#         )
#     plt.plot(data["nirh1"]["ne"].R, data["nirh1"]["ne"].z)
#     plt.title(f"{pulse} Plasma equilibrium")
#     plt.xlabel("R (m)")
#     plt.ylabel("z (m)")
#     plt.axis("scaled")
#     plt.xlim(0, 0.8)
#     plt.ylim(-0.6, 0.6)
#     if savefig:
#         plots.save_figure(fig_name=f"{figname}2D_equilibrium")
#
#     # if save_pickle or write:
#     #     save_to_pickle(pl, raw_data, data, bckc, pulse=pl.pulse, name=run_name)
#
#     #     picklefile = f"/home/marco.sertoli/data/Indica/{pl.pulse}_{run_name}_HDA.pkl"
#     #     pickle.dump([pl, raw_data, data, bckc], open(picklefile, "wb"))
#
#     return pl_all, raw_data, data, bckc_all
#
#
# def xrcs_sensitivity(pulse=9391, write=False):
#     """
#     Test optimisation methods and use of different Te measurements but identical profile shapes
#     on final results (Te and Ti + Wth)
#
#     Test for 9539 saved to 25009539 runs 40, 41, 42, 43
#     """
#
#     # W-line emission moments, n3w PPAC result
#     run_name = "RUN40"
#     descr = f"Moment analysis w-line, Te(n3w)"
#     res = plasma_workflow(
#         pulse=pulse,
#         tstart=0.02,
#         tend=0.12,
#         diagn_ne="smmh1",
#         quant_te="te_n3w",
#         imp_conc=(0.03, 0.001, 0.01),
#         marchuk=True,
#         xrcs_time=True,
#         use_ratios=False,
#         calc_error=False,
#     )
#     pl, raw_data, data, bckc = res
#
#     dt_xrcs = raw_data["xrcs"]["ti_w"].t[1] - raw_data["xrcs"]["ti_w"].t[0]
#     _pl = deepcopy(pl)
#     _bckc = deepcopy(bckc)
#
#     pulse = pl.pulse + 25000000
#     if write:
#         hda_tree.write(
#             pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name
#         )
#
#     # # W-line emission moments, kw PPAC result
#     # run_name = "RUN41"
#     # descr = f"Moment analysis w-line, Te(kw)"
#     # pl = deepcopy(_pl)
#     # bckc = deepcopy(_bckc)
#     # bckc = pl.match_xrcs_temperatures(
#     #     data,
#     #     bckc=bckc,
#     #     quantity_te="te_kw",
#     #     use_ratios=False,
#     #     use_satellites=False,
#     #     calc_error=False,
#     # )
#     # pl, bckc = propagate(
#     #     pl, data, bckc, quant_ar="int_w", cal_ar=0.03, dt=dt_xrcs,
#     # )
#     # if write:
#     #     hda_tree.write(
#     #         pl,
#     #         pulse,
#     #         "HDA",
#     #         data=data,
#     #         bckc=bckc,
#     #         descr=descr,
#     #         run_name=run_name,
#     #     )
#     #
#     # # k/w ratio
#     # run_name = "RUN42"
#     # descr = f"Line ratios, Te(kw)"
#     # pl = deepcopy(_pl)
#     # bckc = deepcopy(_bckc)
#     # bckc = pl.match_xrcs_temperatures(
#     #     data,
#     #     bckc=bckc,
#     #     quantity_te="te_kw",
#     #     use_ratios=True,
#     #     use_satellites=False,
#     #     calc_error=False,
#     # )
#     # pl, bckc = propagate(
#     #     pl, data, bckc, quant_ar="int_w", cal_ar=0.03, dt=dt_xrcs,
#     # )
#     # if write:
#     #     hda_tree.write(
#     #         pl,
#     #         pulse,
#     #         "HDA",
#     #         data=data,
#     #         bckc=bckc,
#     #         descr=descr,
#     #         run_name=run_name,
#     #     )
#
#     # n3/w ratio
#     run_name = "RUN43"
#     descr = f"Line ratios, Te(n3w)"
#     pl = deepcopy(_pl)
#     bckc = deepcopy(_bckc)
#     bckc = pl.match_xrcs_temperatures(
#         data, bckc=bckc, quantity_te="te_n3w", use_ratios=True, calc_error=False,
#     )
#     pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
#     if write:
#         hda_tree.write(
#             pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
#         )
#
#     # n3/w ratio more peaked Te profile
#     run_name = "RUN44"
#     descr = f"Line ratios, Te(n3w), more peaked Te"
#     pl = deepcopy(_pl)
#     bckc = deepcopy(_bckc)
#     bckc = pl.match_xrcs_temperatures(
#         data,
#         bckc=bckc,
#         quantity_te="te_n3w",
#         use_ratios=True,
#         calc_error=False,
#         wped=1.5,
#     )
#     pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
#     if write:
#         hda_tree.write(
#             pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
#         )
#
#     # n3/w ratio more peaked Ne profiles (Nimp following Ne)
#     run_name = "RUN45"
#     descr = f"Line ratios, Te(n3w), more peaked Ne"
#     pl = deepcopy(_pl)
#     bckc = deepcopy(_bckc)
#     pl.Ne_prof.wped = 4
#     pl.Ne_prof.build_profile()
#     pl.Nimp_prof.wped = 4
#     pl.Nimp_prof.build_profile()
#     bckc = pl.match_interferometer(data, bckc=bckc, diagnostic="smmh1", quantity="ne")
#     pl.calc_imp_dens()
#     bckc = pl.match_xrcs_temperatures(
#         data, bckc=bckc, quantity_te="te_n3w", use_ratios=True, calc_error=False,
#     )
#     pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
#     if write:
#         hda_tree.write(
#             pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
#         )
#
#     # n3/w ratio more peaked Ne profiles (Nimp following Ne) and Te
#     run_name = "RUN46"
#     descr = f"Line ratios, Te(n3w), more peaked Te, more peaked Ne"
#     pl = deepcopy(_pl)
#     bckc = deepcopy(_bckc)
#     pl.Ne_prof.wped = 5
#     pl.Ne_prof.build_profile()
#     pl.Nimp_prof.wped = 5
#     pl.Nimp_prof.build_profile()
#     bckc = pl.match_interferometer(data, bckc=bckc, diagnostic="smmh1", quantity="ne")
#     pl.calc_imp_dens()
#     bckc = pl.match_xrcs_temperatures(
#         data,
#         bckc=bckc,
#         quantity_te="te_kw",
#         use_ratios=True,
#         calc_error=False,
#         wped=1.5,
#     )
#     pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
#     if write:
#         hda_tree.write(
#             pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
#         )
#
#     # n3/w ratio EVEN more peaked Ne profiles (Nimp following Ne) and Te
#     run_name = "RUN47"
#     descr = f"Line ratios, Te(n3w), more peaked Te, much more peaked Ne"
#     pl = deepcopy(_pl)
#     bckc = deepcopy(_bckc)
#     pl.Ne_prof.wped = 5
#     pl.Ne_prof.peaking = 1.8
#     pl.Ne_prof.build_profile()
#     pl.Nimp_prof.wped = 5
#     pl.Ne_prof.peaking = 1.8
#     pl.Nimp_prof.build_profile()
#     bckc = pl.match_interferometer(data, bckc=bckc, diagnostic="smmh1", quantity="ne")
#     pl.calc_imp_dens()
#     bckc = pl.match_xrcs_temperatures(
#         data,
#         bckc=bckc,
#         quantity_te="te_kw",
#         use_ratios=True,
#         calc_error=False,
#         wped=1.5,
#     )
#     pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
#     if write:
#         hda_tree.write(
#             pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
#         )
#
#     # n3/w ratio broad Ti
#     run_name = "RUN48"
#     descr = f"Line ratios, Te(n3w), more peaked Te, much more peaked Ne"
#     pl = deepcopy(_pl)
#     bckc = deepcopy(_bckc)
#     bckc = pl.match_interferometer(data, bckc=bckc, diagnostic="smmh1", quantity="ne")
#     pl.calc_imp_dens()
#     bckc = pl.match_xrcs_temperatures(
#         data,
#         bckc=bckc,
#         quantity_te="te_kw",
#         use_ratios=True,
#         calc_error=False,
#         wped=1.5,
#         use_ref=False,
#     )
#     pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
#     if write:
#         hda_tree.write(
#             pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
#         )
#
#     return pl, raw_data, data, bckc
#
#
# def sawtoothing(
#     pulse=9229,  #
#     tstart=0.072,
#     tend=0.078,
#     dt=0.001,
#     t_pre=0.073,
#     t_post=0.075,
#     r_inv=0.1,
#     diagn_ne="smmh1",
#     diagn_te="xrcs",
#     quant_ne="ne",
#     quant_te="te_n3w",
#     quant_ti="ti_w",
#     quant_ar="int_w",
#     main_ion="h",
#     impurities=("c", "ar", "he"),
#     imp_conc=(0.03, 0.001, 0.01),
#     cal_ar=1.0,  # 1.0e13
#     marchuk=True,
#     savefig=False,
#     name="",
#     pl=None,
#     raw_data=None,
#     data=None,
#     bckc=None,
# ):
#     """
#     tests.sawtoothing()
#     tests.sawtoothing(t_pre=0.081, t_post=0.0825, tstart=0.078, tend=0.086)
#     tests.sawtoothing(pulse=9391, t_pre=0.0715, t_post=0.0815, tstart=0.07, tend=0.09)
#
#     pulse=9229
#     tstart=0.072
#     tend=0.078
#     dt=0.001
#     t_pre=0.073
#     t_post=0.075
#     r_inv=0.1
#     diagn_ne="smmh1"
#     diagn_te="xrcs"
#     quant_ne="ne"
#     quant_te="te_n3w"
#     quant_ti="ti_w"
#     quant_ar="int_w"
#     main_ion="h"
#     impurities=("c", "ar", "he")
#     imp_conc=(0.03, 0.001, 0.01)
#     marchuk = True
#     cal_ar=1.
#     calc_error=False
#     """
#
#     if pl is None:
#         raw = ST40data(pulse, tstart - 0.01, tend + 0.01)
#         raw.get_all()
#         raw_data = raw.data
#
#         bckc = {}
#         elements = list(main_ion)
#         elements.extend(list(impurities))
#
#         pl = Plasma(tstart=tstart, tend=tend, dt=dt, elements=elements)
#         data = pl.build_data(raw_data, pulse=pulse)
#
#         for i, elem in enumerate(impurities):
#             if elem in pl.ion_conc.element:
#                 pl.ion_conc.loc[dict(element=elem)] = imp_conc[i]
#
#         # Find Ar density sawtooth crash, assuming Te stays constant
#         if "xrcs" in raw_data:
#             pl.forward_models["xrcs"] = XRCSpectrometer(marchuk=marchuk)
#         if "princeton" in raw_data:
#             pl.forward_models["princeton"] = PISpectrometer()
#
#         pl.set_neutral_density(y1=1.0e15, y0=1.0e9)
#         pl.build_atomic_data()
#         pl.calculate_geometry()
#
#     dt_xrcs = raw_data["xrcs"]["ti_w"].t[1] - raw_data["xrcs"]["ti_w"].t[0]
#
#     # Reference times before and after the crash, inversion radius in rho
#     t_pre = pl.time.values[np.argmin(np.abs(pl.time - t_pre).values)]
#     t_post = pl.time.values[np.argmin(np.abs(pl.time - t_post).values)]
#     t_mid = pl.t[np.abs(pl.t - (t_pre + t_post) / 2.0).argmin()]
#     R_inv = pl.equilibrium.rmag.sel(t=t_post, method="nearest")
#     z_inv = pl.equilibrium.zmag.sel(t=t_post, method="nearest") + r_inv
#     rho_inv, _, _ = pl.equilibrium.flux_coords(R_inv, z_inv, t=t_post)
#
#     # Find electron density sawtooth crash
#     ne_pre_data = data["smmh1"]["ne"].sel(t=t_pre).values
#     ne_post_data = data["smmh1"]["ne"].sel(t=t_post).values
#
#     # Test different peaking factors to match both pre and post crash profiles
#     pl.Ne_prof.y1 = 0.5e19
#     pl.Ne_prof.wcenter = rho_inv.values / 2.0
#     pl.Ne_prof.wped = 5
#     pl.Ne_prof.peaking = 1.0
#     pl.Ne_prof.build_profile()
#     bckc = pl.match_interferometer(
#         data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne,
#     )
#     pl.calc_imp_dens()
#     volume = pl.volume.sel(t=t_post)
#
#     scan = np.linspace(1.0, 2.5, 21)
#     Ne_pre, Ne_post = [], []
#     ne_pre_bckc, ne_post_bckc = [], []
#     for s in scan:
#         pre = deepcopy(pl.Ne_prof)
#         pre.peaking = s
#         pre.build_profile()
#         pl.el_dens.loc[dict(t=t_pre)] = pre.yspl.values
#         pre.y0 *= (
#             ne_pre_data / pl.calc_ne_los_int(data[diagn_ne][quant_ne], t=t_pre)
#         ).values
#         pre.build_profile()
#         pl.el_dens.loc[dict(t=t_pre)] = pre.yspl.values
#         ne_pre_bckc.append(pl.calc_ne_los_int(data[diagn_ne][quant_ne], t=t_pre).values)
#         Ne_pre.append(deepcopy(pre.yspl.values))
#         pre_crash = pl.el_dens.sel(t=t_pre)
#         post_crash = ph.sawtooth_crash(
#             pre_crash.rho_poloidal, pre_crash.values, volume, rho_inv
#         )
#         Ne_post.append(deepcopy(post_crash))
#         pl.el_dens.loc[dict(t=t_post)] = post_crash
#         ne_post_bckc.append(
#             pl.calc_ne_los_int(data[diagn_ne][quant_ne], t=t_post).values
#         )
#
#     ne_post_bckc = np.array(ne_post_bckc)
#
#     plt.figure()
#     raw_data["smmh1"]["ne"].plot()
#     data["smmh1"]["ne"].plot(linewidth=3)
#     ylim = plt.ylim()
#     plt.vlines(t_post, ylim[0], ylim[1], color="black", linestyle="dashed")
#     plt.plot(t_pre, ne_pre_data, marker="o", color="black")
#     plt.plot(t_post, ne_post_data, marker="o", color="red")
#
#     colors = cm.rainbow(np.linspace(0, 1, len(scan)))
#     for i, s, in enumerate(scan):
#         plt.plot(t_post, ne_post_bckc[i], "x", color=colors[i])
#
#     if savefig:
#         figname = plots.get_figname(pulse=pl.pulse, name=name)
#         plots.save_figure(fig_name=f"{figname}data_electron_density_peaking_scan")
#
#     ind = np.argmin(np.abs(ne_post_data - ne_post_bckc))
#     plt.figure()
#     for i, s, in enumerate(scan):
#         plt.plot(pl.rho, Ne_pre[i], color=colors[i], alpha=0.5)
#         plt.plot(pl.rho, Ne_post[i], color=colors[i], linestyle="dashed", alpha=0.5)
#
#     plt.plot(pl.rho, Ne_pre[ind], color="black", marker="o")
#     plt.plot(pl.rho, Ne_post[ind], color=colors[ind], linestyle="dashed", marker="o")
#
#     if savefig:
#         figname = plots.get_figname(pulse=pl.pulse, name=name)
#         plots.save_figure(fig_name=f"{figname}profiles_electron_density_peaking_scan")
#
#     # Fix electron density crash to best matching
#     Ne_pre = DataArray(Ne_pre[ind], coords=[("rho_poloidal", pl.rho)])
#     Ne_post = DataArray(Ne_post[ind], coords=[("rho_poloidal", pl.rho)])
#     pl.el_dens = xr.where(pl.el_dens.t <= t_mid, Ne_pre, Ne_post)
#
#     pl.calc_imp_dens()
#
#     # Build tempertaure profiles to match XRCS using standard shapes
#     bckc = pl.match_xrcs_temperatures(
#         data,
#         bckc=bckc,
#         diagnostic=diagn_te,
#         quantity_te=quant_te,
#         quantity_ti=quant_ti,
#         use_ratios=True,
#         time=[t_pre, t_post],
#     )
#     t_xrcs = raw_data["xrcs"]["int_w"].t
#     t_pre_xrcs = np.nanmax(xr.where(t_xrcs < t_pre, t_xrcs, np.nan).values)
#     t_post_xrcs = np.nanmin(xr.where(t_xrcs > t_post, t_xrcs, np.nan).values)
#     t_pre_xrcs = pl.time.values[np.argmin(np.abs(pl.time - t_pre_xrcs).values)]
#     t_post_xrcs = pl.time.values[np.argmin(np.abs(pl.time - t_post_xrcs).values)]
#     pl.el_temp = xr.where(
#         pl.el_temp.t <= t_mid, pl.el_temp.sel(t=t_pre), pl.el_temp.sel(t=t_post)
#     )
#     for elem in pl.elements:
#         ion_temp = pl.ion_temp.sel(element=elem)
#         ion_temp = xr.where(
#             ion_temp.t <= t_mid, ion_temp.sel(t=t_pre), ion_temp.sel(t=t_post)
#         )
#         pl.ion_temp.loc[dict(element=elem)] = ion_temp.values
#
#     int_pre_data = data["xrcs"]["int_w"].sel(t=t_pre_xrcs, method="nearest").values
#     int_post_data = data["xrcs"]["int_w"].sel(t=t_post_xrcs, method="nearest").values
#
#     attrs = ["y0", "y1", "yend", "wped", "wcenter"]
#     for a in attrs:
#         setattr(pl.Nimp_prof, a, getattr(pl.Ne_prof, a))
#
#     wcentre = 0.3
#     wped = 1
#     pl.Nimp_prof.peaking = 1.0
#     pl.Nimp_prof.y1 = pl.Nimp_prof.y0 / 2.0
#     pl.Nimp_prof.yend = pl.Nimp_prof.y1
#     pl.Nimp_prof.wcenter = wcentre
#     pl.Nimp_prof.wped = wped
#     pl.Nimp_prof.build_profile()
#     Nimp_first = deepcopy(pl.Nimp_prof)
#     volume = pl.volume.sel(t=t_post)
#
#     pl.calc_fz_lz()
#     pl.calc_meanz()
#     pl.calc_imp_dens()
#
#     Nimp_pre, Nimp_post = [], []
#     int_pre_bckc, int_post_bckc = [], []
#     scan = np.linspace(1.0, 5, 21) ** 2
#     for s in scan:
#         # Pre crash profile
#         pl.Nimp_prof = deepcopy(Nimp_first)
#         pl.Nimp_prof.peaking = s
#         pl.Nimp_prof.y1 /= s
#         pl.Nimp_prof.yend /= s
#         pl.Nimp_prof.wcenter = 0.12
#         pl.Nimp_prof.build_profile()
#         pl.ion_dens.loc[dict(element="ar", t=t_pre)] = pl.Nimp_prof.yspl.values
#         bckc = pl.match_xrcs_intensity(
#             data,
#             bckc=bckc,
#             diagnostic="xrcs",
#             quantity=quant_ar,
#             time=[t_pre],
#             cal=cal_ar,
#             dt=dt_xrcs,
#         )
#         int_pre_bckc.append(deepcopy(bckc["xrcs"][quant_ar].sel(t=t_pre).values))
#         pre_crash = pl.ion_dens.sel(element="ar", t=t_pre)
#         Nimp_pre.append(deepcopy(pre_crash.values))
#         post_crash = ph.sawtooth_crash(
#             pre_crash.rho_poloidal, pre_crash.values, volume, rho_inv
#         )
#         Nimp_post.append(deepcopy(post_crash))
#         pl.ion_dens.loc[dict(element="ar", t=t_post)] = post_crash
#         bckc = pl.match_xrcs_intensity(
#             data,
#             bckc=bckc,
#             diagnostic="xrcs",
#             quantity=quant_ar,
#             time=[t_post],
#             scale=False,
#             cal=cal_ar,
#             dt=dt_xrcs,
#         )
#         int_post_bckc.append(deepcopy(bckc["xrcs"][quant_ar].sel(t=t_post).values))
#
#     int_pre_bckc = np.array(int_pre_bckc)
#     int_post_bckc = np.array(int_post_bckc)
#
#     plt.figure()
#     raw_data["xrcs"]["int_w"].plot()
#     data["xrcs"]["int_w"].plot(linewidth=3)
#     ylim = plt.ylim()
#     plt.vlines(t_post_xrcs, ylim[0], ylim[1], color="black", linestyle="dashed")
#     plt.plot(t_pre_xrcs, int_pre_data, marker="o", color="black")
#     plt.plot(t_post_xrcs, int_post_data, marker="o", color="red")
#
#     colors = cm.rainbow(np.linspace(0, 1, len(scan)))
#     for i, s, in enumerate(scan):
#         plt.plot(t_post_xrcs, int_pre_bckc[i], "d", color=colors[i])
#         plt.plot(t_post_xrcs, int_post_bckc[i], "x", color=colors[i])
#
#     if savefig:
#         figname = plots.get_figname(pulse=pl.pulse, name=name)
#         plots.save_figure(fig_name=f"{figname}data_XRCS_argon_density_peaking_scan")
#
#     plt.figure()
#     ind = np.argmin(np.abs(int_post_data - int_post_bckc))
#     for i, s, in enumerate(scan):
#         plt.plot(pl.rho, Nimp_pre[i], color=colors[i], alpha=0.5)
#         plt.plot(pl.rho, Nimp_post[i], color=colors[i], linestyle="dashed", alpha=0.5)
#
#     plt.plot(pl.rho, Nimp_pre[ind], color="black", marker="o")
#     plt.plot(pl.rho, Nimp_post[ind], color=colors[ind], linestyle="dashed", marker="o")
#
#     if savefig:
#         figname = plots.get_figname(pulse=pl.pulse, name=name)
#         plots.save_figure(fig_name=f"{figname}profiles_XRCS_argon_density_peaking_scan")
#
#     # Fix electron density crash to best matching
#     Nimp_pre = DataArray(Nimp_pre[ind], coords=[("rho_poloidal", pl.rho)])
#     Nimp_post = DataArray(Nimp_post[ind], coords=[("rho_poloidal", pl.rho)])
#     Nimp = pl.ion_dens.sel(element="ar")
#     Nimp = xr.where(Nimp.t <= t_mid, Nimp_pre, Nimp_post)
#     pl.ion_dens.loc[dict(element="ar")] = Nimp.values
#
#     # Build tempertaure profiles to match XRCS using standard shapes
#     bckc = pl.match_xrcs_temperatures(
#         data,
#         bckc=bckc,
#         diagnostic=diagn_te,
#         quantity_te=quant_te,
#         quantity_ti=quant_ti,
#         use_ratios=True,
#     )
#     pl.calc_fz_lz()
#     pl.calc_meanz()
#     bckc = pl.match_xrcs_intensity(
#         data,
#         bckc=bckc,
#         diagnostic="xrcs",
#         quantity=quant_ar,
#         scale=False,
#         cal=cal_ar,
#         dt=dt_xrcs,
#     )
#     pl.calc_main_ion_dens()
#     pl.calc_zeff()
#     pl.calc_rad_power()
#
#     bckc = pl.calc_pressure()
#     bckc = pl.interferometer(data, bckc=bckc)
#     bckc = pl.bremsstrahlung(data, bckc=bckc)
#
#     plots.compare_data_bckc(
#         data, bckc, raw_data=raw_data, pulse=pl.pulse, savefig=savefig, name=name,
#     )
#     plots.profiles(pl, data=data, savefig=savefig, name=name)
#     plots.time_evol(pl, data, bckc=bckc, savefig=savefig, name=name)
#
#     return pl, raw_data, data, bckc
#
#
# def compare_astra(pulse=8574, tstart=0.02, tend=0.14, revision=105, interf="nirh1"):
#     pulse = 8574
#     tstart = 0.02
#     tend = 0.14
#     revision = "105"
#     reader = ST40Reader(int(pulse + 25.0e6), tstart, tend, tree="astra")
#     astra = reader.get("", "astra", revision)
#     astra["ne"] *= 1.0e19
#     astra["ni"] *= 1.0e19
#     astra["te"] *= 1.0e3
#     astra["ti"] *= 1.0e3
#     # rho_poloidal = astra["p"].rho_poloidal
#     # rho_toroidal = astra["ne"].rho_toroidal
#
#     time = astra["te"].t
#     tstart, tend = time[0], time[-1]
#     dt = time[1] - time[0]
#     pl, raw_data, data, bckc = plasma_workflow(
#         pulse=pulse, tstart=tstart, tend=tend, dt=dt
#     )
#
#
# def best_astra(
#     pulse=8383, tstart=0.02, tend=0.12, hdarun=None, write=False, force=False
# ):
#     """
#     Best profile shapes from ASTRA runs of 8383 applied to database
#     """
#     ohmic_pulses = [8385, 8386, 8387, 8390, 8405, 8458]  # 8401
#     nbi_pulses = [8338, 8373, 8374, 8574, 8575, 8582, 8583, 8597, 8598, 8599]  #
#
#     pulses = np.sort(np.concatenate((np.array(ohmic_pulses), np.array(nbi_pulses))))
#     if pulse is not None:
#         pulses = [pulse]
#     for pulse in pulses:
#         interf = "nirh1"
#         hdarun = HDArun(pulse=pulse, interf=interf, tstart=tstart, tend=tend)
#
#         # Rebuild temperature profiles
#         hdarun.profiles_nbi()
#         profs_spl = Plasma_profs(hdarun.data.time)
#
#         # Rescale to match XRCS measurements
#         hdarun.data.match_xrcs(profs_spl=profs_spl)
#
#         # Recalculate average charge, dilution, Zeff, total pressure
#         hdarun.data.calc_meanz()
#         hdarun.data.calc_main_ion_dens(fast_dens=False)
#         hdarun.data.impose_flat_zeff()
#         hdarun.data.calc_main_ion_dens(fast_dens=False)
#         hdarun.data.calc_zeff()
#         hdarun.data.calc_pressure()
#
#         descr = f"Best profile shapes from ASTRA {pulse}, c_C=3%"
#         run_name = "RUN30"
#         plt.close("all")
#         hdarun.plot()
#         if write:
#             hdarun.write(hdarun.data, descr=descr, run_name=run_name, force=force)
#         else:
#             return hdarun
#
#
# def scan_profile_shape(pulse=8383, hdarun=None, write=False):
#     """
#     Fix edge plasma parameters (rho > 0.8) and scan profile shapes
#     """
#
#     interf = "nirh1"
#     if hdarun is None:
#         hdarun = HDArun(pulse=pulse, interf=interf, tstart=0.02, tend=0.1)
#
#     # Temperature profile shape scan, flat density
#     hdarun.profiles_ohmic()
#
#     te_flat = deepcopy(hdarun)
#     te_peak1 = deepcopy(hdarun)
#     te_peak2 = deepcopy(hdarun)
#
#     profs_spl = Plasma_profs(te_flat.data.time)
#     te_flat.data.match_xrcs(profs_spl=profs_spl)
#     te_flat.data.calc_pressure()
#     descr = "Flat density, flat temperature, c_C=3%"
#     run_name = "RUN10"
#     if write == True:
#         te_flat.write(te_flat.data, descr=descr, run_name=run_name)
#
#     profs_spl.el_temp.scale(2.0, dim_lim=(0, 0))
#     te_peak1.data.match_xrcs(profs_spl=profs_spl)
#     te_peak1.data.calc_pressure()
#     descr = "Flat density, peaked temperature, c_C=3%"
#     run_name = "RUN11"
#     if write == True:
#         te_peak1.write(te_peak1.data, descr=descr, run_name=run_name)
#
#     profs_spl.el_temp.scale(0.5, dim_lim=(0.7, 0.98))
#     profs_spl.ion_temp.scale(2.0, dim_lim=(0, 0))
#     profs_spl.ion_temp.scale(0.5, dim_lim=(0.7, 0.98))
#     te_peak2.data.match_xrcs(profs_spl=profs_spl)
#     te_peak2.data.calc_pressure()
#     descr = "Flat density, very peaked temperature, c_C=3%"
#     run_name = "RUN12"
#     if write == True:
#         te_peak2.write(te_peak2.data, descr=descr, run_name=run_name)
#
#     flat_dens = {"te_flat": te_flat, "te_peak1": te_peak1, "te_peak2": te_peak2}
#
#     # Peaked density
#     hdarun.profiles_nbi()
#
#     te_flat = deepcopy(hdarun)
#     te_peak1 = deepcopy(hdarun)
#     te_peak2 = deepcopy(hdarun)
#
#     profs_spl = Plasma_profs(te_flat.data.time)
#     te_flat.data.match_xrcs(profs_spl=profs_spl)
#     te_flat.data.calc_pressure()
#     descr = "Peaked density, flat temperature, c_C=3%"
#     run_name = "RUN20"
#     if write == True:
#         te_flat.write(te_flat.data, descr=descr, run_name=run_name)
#
#     profs_spl.el_temp.scale(2.0, dim_lim=(0, 0))
#     te_peak1.data.match_xrcs(profs_spl=profs_spl)
#     te_peak1.data.calc_pressure()
#     descr = "Peaked density, peaked temperature, c_C=3%"
#     run_name = "RUN21"
#     if write == True:
#         te_peak1.write(te_peak1.data, descr=descr, run_name=run_name)
#
#     profs_spl.el_temp.scale(0.5, dim_lim=(0.7, 0.98))
#     profs_spl.ion_temp.scale(2.0, dim_lim=(0, 0))
#     profs_spl.ion_temp.scale(0.5, dim_lim=(0.7, 0.98))
#     te_peak2.data.match_xrcs(profs_spl=profs_spl)
#     te_peak2.data.calc_pressure()
#     descr = "Peaked density, very peaked temperature, c_C=3%"
#     run_name = "RUN22"
#     if write == True:
#         te_peak2.write(te_peak2.data, descr=descr, run_name=run_name)
#
#     peaked_dens = {"te_flat": te_flat, "te_peak1": te_peak1, "te_peak2": te_peak2}
#
#     if not write:
#         return flat_dens, peaked_dens
#
#
# def ohmic_pulses(write=False, interf="smmh1", match_kinetic=False):
#     # pulses = [8385, 8386, 8387, 8390, 8401, 8405, 8458]
#     for pulse in pulses:
#         hdarun = HDArun(pulse=pulse, interf=interf, tstart=0.02, tend=0.1)
#         hdarun.profiles_ohmic()
#         if match_kinetic:
#             hdarun.data.calc_pressure()
#             descr = "New profile shapes, match kinetic profiles only, c_C=3%"
#             run_name = "RUN01"
#         else:
#             hdarun.match_energy()
#             descr = "New profile shapes, adapt Ne to match Wmhd, c_C=3%"
#             run_name = "RUN05"
#         if write == True:
#             hdarun.write(hdarun.bckc, descr=descr, run_name=run_name)
#         else:
#             hdarun.plot()
#
#     return hdarun
#
#
# def NBI_pulses(write=False, interf="smmh1", match_kinetic=False):
#     pulses = [8338, 8574, 8575, 8582, 8583, 8597, 8598, 8599]
#     interf = ["nirh1"] * len(pulses)
#     for i, pulse in enumerate(pulses):
#         plt.close("all")
#         hdarun = HDArun(
#             pulse=pulse, interf=interf[i], tstart=0.015, tend=0.14, dt=0.015
#         )
#         hdarun.profiles_nbi()
#         if match_kinetic:
#             hdarun.data.calc_pressure()
#             descr = "New profile shapes, match kinetic measurements only, c_C=3%"
#             run_name = "RUN01"
#         else:
#             hdarun.match_energy()
#             descr = "New profile shapes, adapt Ne to match Wmhd, c_C=3%"
#             run_name = "RUN05"
#         if write == True:
#             hdarun.write(hdarun.data, descr=descr, run_name=run_name)
#         else:
#             hdarun.plot()
#             _ = input("press...")
#
#     return hdarun
#
#
# def test_low_edge_temperature(hdarun, zeff=False):
#
#     # low temperature edge
#     hdarun.initialize_bckc()
#     te_0 = 1.0e3
#     hdarun.bckc.profs.te = hdarun.bckc.profs.build_temperature(
#         y_0=te_0,
#         y_ped=te_0 / 15.0,
#         x_ped=0.9,
#         w_core=0.2,
#         datatype=("temperature", "electron"),
#     )
#     hdarun.bckc.profs.te /= hdarun.bckc.profs.te.max()
#     elements = hdarun.bckc.elements
#     main_ion = hdarun.bckc.main_ion
#     for t in hdarun.bckc.time:
#         te_0 = hdarun.bckc.el_temp.sel(t=t).sel(rho_poloidal=0).values
#         hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values
#         ti_0 = (
#             hdarun.bckc.ion_temp.sel(element=main_ion)
#             .sel(t=t)
#             .sel(rho_poloidal=0)
#             .values
#         )
#         for elem in elements:
#             hdarun.bckc.ion_temp.loc[dict(t=t, element=elem)] = (
#                 hdarun.bckc.profs.te * ti_0
#             ).values
#
#     hdarun.bckc.match_xrcs()
#     hdarun.bckc.simulate_spectrometers()
#
#     # hdarun.recover_zeff(optimize="density")
#
#     hdarun.bckc.propagate_parameters()
#     # hdarun.recover_density()
#
#     hdarun.plot()
#
#
# def rabbit_ears(hdarun: HDArun):
#
#     hdarun.initialize_bckc()
#     ne_0 = hdarun.bckc.profs.ne.sel(rho_poloidal=0).values
#     hdarun.bckc.profs.ne = hdarun.bckc.profs.build_density(
#         x_0=0.7,
#         y_0=ne_0,
#         y_ped=ne_0 / 4.0,
#         x_ped=0.95,
#         w_core=0.1,
#         datatype=("density", "electron"),
#     )
#
#     for t in hdarun.bckc.time:
#         hdarun.bckc.el_dens.loc[dict(t=t)] = hdarun.bckc.profs.ne.values
#     hdarun.bckc.match_interferometer(interf)
#
#     # hdarun.recover_density()
#
#     hdarun.plot()
#
#
# def test_peaked_profiles(hdarun, zeff=False):
#     hdarun.initialize_bckc()
#     hdarun.recover_density()
#     if zeff:
#         hdarun.recover_zeff(optimize="density")
#     hdarun.bckc.simulate_spectrometers()
#     broad = hdarun.bckc
#
#     # Peaked profiles
#     hdarun.initialize_bckc()
#     te_0 = 1.0e3
#     hdarun.bckc.profs.te = hdarun.bckc.profs.build_temperature(
#         y_0=te_0,
#         y_ped=te_0 / 15.0,
#         x_ped=0.9,
#         w_core=0.3,
#         datatype=("temperature", "electron"),
#     )
#     hdarun.bckc.profs.te /= hdarun.bckc.profs.te.max()
#
#     ne_0 = 5.0e19
#     hdarun.bckc.profs.ne = hdarun.bckc.profs.build_temperature(
#         y_0=ne_0,
#         y_ped=ne_0 / 15.0,
#         x_ped=0.9,
#         w_core=0.3,
#         datatype=("density", "electron"),
#     )
#     hdarun.bckc.profs.ne /= hdarun.bckc.profs.ne.max()
#     for t in hdarun.bckc.time:
#         te_0 = hdarun.bckc.el_temp.sel(t=t).sel(rho_poloidal=0).values
#         hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values
#
#         ne_0 = hdarun.bckc.el_dens.sel(t=t).sel(rho_poloidal=0).values
#         hdarun.bckc.el_dens.loc[dict(t=t)] = (hdarun.bckc.profs.ne * ne_0).values
#
#     hdarun.bckc.build_current_density()
#     hdarun.recover_density()
#     if zeff:
#         hdarun.recover_zeff(optimize="density")
#     hdarun.bckc.simulate_spectrometers()
#     peaked = hdarun.bckc
#
#     HDAplot(broad, peaked)
#
#
# def test_current_density(hdarun):
#     """Trust all measurements, find shape to explain data"""
#
#     # L-mode profiles
#
#     # Broad current density
#     hdarun.initialize_bckc()
#     hdarun.bckc.build_current_density(sigm=0.8)
#     hdarun.recover_density()
#     hdarun.recover_zeff(optimize="density")
#     broad = deepcopy(hdarun.bckc)
#
#     # Peaked current density
#     hdarun.initialize_bckc()
#     hdarun.bckc.build_current_density(sigm=0.2)
#     hdarun.recover_density()
#     hdarun.recover_zeff(optimize="density")
#     peaked = deepcopy(hdarun.bckc)
#
#     HDAplot(broad, peaked)
