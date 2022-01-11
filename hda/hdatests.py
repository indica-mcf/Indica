from copy import deepcopy
from matplotlib import cm, rcParams

from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import pickle
import xarray as xr
from xarray import DataArray

from hda.hdaplot import HDAplot
from hda.hdaworkflow import HDArun
from hda.spline_profiles import Plasma_profs
from hda.read_st40 import ST40data
from hda.plasma import Plasma
import hda.plots as plots
import hda.hda_tree as hda_tree
import hda.physics as ph
import hda.profiles as profiles
from indica.readers import ST40Reader

from hda.diagnostics.spectrometer import XRCSpectrometer
from hda.diagnostics.PISpectrometer import PISpectrometer

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
    run_name="RUN40",
    descr=f"New profile shapes and ionisation balance",  # descr = "Experimental evolution of the Ar concentration"
    name="",
    xrcs_time=True,
    use_ratios=True,
    calc_error=False,
    sxr=False,
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
    raw.get_all(sxr=sxr)  # smmh1_rev=2)
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
        pl.forward_models["xrcs"] = XRCSpectrometer(marchuk=marchuk)
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

    pl.calc_fz_lz()
    pl.calc_meanz()

    # Default impurity concentrations
    pl.calc_imp_dens()

    # Ar density from intensity of w line
    bckc = pl.match_xrcs_intensity(
        data, bckc=bckc, diagnostic="xrcs", quantity=quant_ar, cal=cal_ar, dt=dt_xrcs,
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
            data, bckc, raw_data=raw_data, pulse=pl.pulse, savefig=savefig, name=name,
        )
        plots.profiles(pl, bckc=bckc, savefig=savefig, name=name)
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
        picklefile = f"/home/marco.sertoli/data/Indica/{pl.pulse}_{run_name}_HDA.pkl"
        pickle.dump([pl, raw_data, data, bckc], open(picklefile, "wb"))

    return pl, raw_data, data, bckc


def propagate(pl, raw_data, data, bckc, quant_ar="int_w", cal_ar=1):
    dt_xrcs = (raw_data["xrcs"]["ti_w"].t[1] - raw_data["xrcs"]["ti_w"].t[0]).values
    pl.calc_meanz()
    pl.calc_imp_dens()
    pl.match_xrcs_intensity(
        data, bckc=bckc, diagnostic="xrcs", quantity=quant_ar, cal=cal_ar, dt=dt_xrcs,
    )
    pl.calc_main_ion_dens()
    pl.calc_zeff()
    pl.calc_pressure()
    pl.calc_rad_power()
    pl.interferometer(data, bckc=bckc)
    pl.bremsstrahlung(data, bckc=bckc)
    return pl, bckc


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
    write=False,
    save_pickle=False,
    modelling=True,
    res=None,
    force=True,
    sxr=False,
    main_ion="h",
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
            marchuk=True,
            xrcs_time=False,
            use_ratios=True,
            calc_error=False,
            cal_ar=1,
            sxr=sxr,
            main_ion=main_ion,
        )
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
                pl.Ti_prof = deepcopy(Ti)
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
                    run_name = f"RUN{run_tmp}"
                    descr = f"{kTe} Te, {kTi} Ti, {kNe} Ne, {kNimp} Nimp"
                    print(f"\n{descr}\n")
                    run_dict[run_name] = descr

                    pl.match_interferometer(
                        data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne
                    )
                    pl.calc_imp_dens()

                    pl.Nimp_prof.yspl.plot()
                    pl.ion_dens.sel(element="ar").sel(
                        t=pl.time.mean(), method="nearest"
                    ).plot()

                    # if iteration > 3:
                    #     return

                    pl.match_xrcs_temperatures(
                        data,
                        bckc=bckc,
                        diagnostic="xrcs",
                        quantity_te=quant_te,
                        quantity_ti=quant_ti,
                        use_ratios=True,
                        calc_error=False,
                    )
                    propagate(pl, raw_data, data, bckc, quant_ar="int_w")
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
                            force=force,
                        )

                    if save_pickle or write:
                        picklefile = f"/home/marco.sertoli/data/Indica/{pl.pulse}_{run_name}_HDA.pkl"
                        pickle.dump([pl, raw_data, data, bckc], open(picklefile, "wb"))

                    if run_tmp > 80:
                        break

                    iteration += 1

    elem = "ar"
    t = pl.time.values[int(len(pl.time) / 2.0)]

    cols = cm.rainbow(np.linspace(0, 1, len(pl_dict.keys())))

    colors = {}
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
        colors[run_name] = cols[len(ne0)]
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

    run_name = f"RUN{run}"
    descr = f"Average over runs {runs[0]}-{runs[-1]}"
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
        )

    if save_pickle or write:
        picklefile = f"/home/marco.sertoli/data/Indica/{pl.pulse}_{run_name}_HDA.pkl"
        pickle.dump([pl, raw_data, data, bckc], open(picklefile, "wb"))

    plt.figure()
    for run_name, pl in pl_dict.items():
        pl.el_temp.sel(t=t).plot(color=colors[run_name])

    plt.figure()
    for run_name, pl in pl_dict.items():
        pl.ion_temp.sel(t=t, element=elem).plot(color=colors[run_name])

    plt.figure()
    for run_name, pl in pl_dict.items():
        pl.el_dens.sel(t=t).plot(color=colors[run_name])

    plt.figure()
    for run_name, pl in pl_dict.items():
        pl.ion_dens.sel(t=t, element=elem).plot(color=colors[run_name])

    plt.figure()
    for run_name, pl in pl_dict.items():
        pl.el_temp.sel(rho_poloidal=0).plot(color=colors[run_name])
        pl.ion_temp.sel(rho_poloidal=0, element=elem).plot(
            color=colors[run_name], linestyle="dashed"
        )
    plt.ylim(0,)
    plt.title("Te(0) and Ti(0)")
    plt.ylabel("(eV)")

    plt.figure()
    for run_name, pl in pl_dict.items():
        pl.el_dens.sel(rho_poloidal=0).plot(color=colors[run_name])
    plt.ylim(0,)
    plt.title("Ne(0)")
    plt.ylabel("(m$^{-3}$)")

    return pl_dict, raw_data, data, bckc_dict


def vertical_displacement(
    pulse=8804,
    tstart=0.05,
    tend=0.12,
    dt=0.01,
    diagn_ne="nirh1_bin",
    quant_ne="ne",
    save_pickle=False,
    savefig=False,
    name="vertical_displacement",
    perc_err=0.05,
    y1=1.e19,
    nsteps=6,
    ref_pulse=None,
):
    """
    # These are really good pulses to compare, with constant flat-top-parameters
    # and goo geometry scan --> flat density profile
    res = tests.vertical_displacement(
        pulse=8804,
        tstart=0.05,
        tend=0.12,
        dt=0.01,
        diagn_ne="nirh1_bin",
        quant_ne="ne",
    )

    res = tests.vertical_displacement(
        pulse=8806,
        tstart=0.07,
        tend=0.13,
        dt=0.01,
        diagn_ne="nirh1_bin",
        quant_ne="ne",
    )

    res = tests.vertical_displacement(
        pulse=8807,
        tstart=0.07,
        tend=0.12,
        dt=0.015,
        diagn_ne="nirh1_bin",
        quant_ne="ne",
    )

    # The following two are reference and z-shift pulse, but they have different RFX
    # waveforms and RFX switch-off during the ramp so are not good to compare
    res = tests.vertical_displacement(
        pulse=9651,
        tstart=0.08,
        tend=0.12,
        dt=0.012,
        diagn_ne="nirh1",
        quant_ne="ne",
        y1=0.1e19,
    )

    res = tests.vertical_displacement(
        pulse=9652,
        tstart=0.08,
        tend=0.12,
        dt=0.012,
        diagn_ne="nirh1",
        quant_ne="ne",
        y1=0.1e19,
    )
    """

    def line(x, intercept, slope):
        return intercept + x * slope

    main_ion = "h"
    impurities = ("c", "ar", "he")

    raw = ST40data(pulse, tstart - dt * 2, tend + dt * 2)
    raw.get_all()
    raw_data = raw.data

    bckc = {}
    elements = list(main_ion)
    elements.extend(list(impurities))

    pl = Plasma(tstart=tstart, tend=tend, dt=dt, elements=elements)
    data = pl.build_data(raw_data, pulse=pulse)

    Ne = deepcopy(pl.Ne_prof)
    wped_all = np.linspace(12, 4, nsteps) # [12, 6, 5, 4]
    peaking_all = np.linspace(1.05, 3, nsteps) # [1.05, 1.3, 2.0, 4]
    colors = cm.turbo(np.linspace(0, 1, nsteps))

    Ne_all = []
    pl_all = []
    bckc_all = []
    for wped, peaking in zip(wped_all, peaking_all):
        Ne.wped = wped
        Ne.y1 = y1
        Ne.peaking = peaking
        Ne.build_profile()
        Ne_all.append(deepcopy(Ne))

        pl.Ne_prof = deepcopy(Ne)

        bckc = pl.match_interferometer(
            data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne
        )

        bckc = pl.interferometer(data, bckc=bckc)
        bckc = pl.bremsstrahlung(data, bckc=bckc)

        pl_all.append(deepcopy(pl))
        bckc_all.append(deepcopy(bckc))

    figname = plots.get_figname(pulse=pulse, name=name)
    cmap = cm.rainbow
    colors_t = cmap(np.linspace(0, 1, len(pl.time)))

    plt.figure()
    t = pl.t[int(len(pl.t)/2)]
    for pl, c in zip(pl_all, colors):
        (pl.el_dens.sel(t=t)/1.e19).plot(color=c)
    plt.title(f"{pulse} Density profiles t = {t:.3f}")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(10$^{19}$ m$^{-3}$)")
    plt.legend()
    if savefig:
        plots.save_figure(fig_name=f"{figname}profiles_electron_densities")

    plt.figure()
    raw = raw_data[diagn_ne][quant_ne]
    value = data[diagn_ne][quant_ne]
    error = value.attrs["error"]
    print(error)
    (raw/1.e19).plot()
    plt.plot(value.t, (value)/1.e19, "-o", color="red")
    plt.fill_between(value.t, (value - error) / 1.e19, (value + error) / 1.e19, alpha=0.5, color="red")
    plt.title(f"{pulse} {diagn_ne.upper()} electron density measurement")
    plt.xlabel("Time (s)")
    plt.ylabel("(10$^{19}$ m$^{-2}$)")
    plt.legend()
    if savefig:
        plots.save_figure(fig_name=f"{figname}data_density_measurement")

    plt.figure()
    for pl, c in zip(pl_all, colors):
        x = pl.time
        y = pl.el_dens.sel(rho_poloidal=0)/1.e19

        pfit = np.polyfit(x.values, y.values, 1)
        slope, intercept = pfit

        yfit = line(x, intercept, slope)
        yfit.plot(linestyle="dashed", color=c)
        residuals = y - yfit
        chi_sq = np.sum((residuals / (y * perc_err)) ** 2)
        plt.plot(x, y, label=f"slope = {slope:.2f}", color=c)
        for t, c in zip(pl.time, colors_t):
            plt.errorbar(
                t, y.sel(t=t), perc_err * y.sel(t=t), alpha=0.8, marker="o", color=c
            )

    plt.title(f"{pulse} Central electron density")
    plt.xlabel("Time (s)")
    plt.ylabel("(10$^{19}$ m$^{-3}$)")
    plt.legend()
    if savefig:
        plots.save_figure(fig_name=f"{figname}time_evol_central_densities")

    plt.figure()
    R = pl.equilibrium.rho.R
    z = pl.equilibrium.rho.z
    vmin = np.linspace(1, 0, len(pl.time))
    for i, t in enumerate(pl.time):
        rho = pl.equilibrium.rho.sel(t=t, method="nearest")
        plt.contour(
            R,
            z,
            rho,
            levels=[1.0],
            alpha=0.5,
            cmap=cmap,
            vmin=vmin[i],
            vmax=vmin[i] + 1,
        )
        plt.plot(
            pl.equilibrium.rmag.sel(t=t, method="nearest"),
            pl.equilibrium.zmag.sel(t=t, method="nearest"),
            color=colors_t[i],
            marker="o",
            alpha=0.5,
        )
    plt.plot(data["nirh1"]["ne"].R, data["nirh1"]["ne"].z)
    plt.title(f"{pulse} Plasma equilibrium")
    plt.xlabel("R (m)")
    plt.ylabel("z (m)")
    plt.axis("scaled")
    plt.xlim(0, 0.8)
    plt.ylim(-0.6, 0.6)
    if savefig:
        plots.save_figure(fig_name=f"{figname}2D_equilibrium")

    # if save_pickle or write:
    #     picklefile = f"/home/marco.sertoli/data/Indica/{pl.pulse}_{run_name}_HDA.pkl"
    #     pickle.dump([pl, raw_data, data, bckc], open(picklefile, "wb"))

    return pl_all, raw_data, data, bckc_all


def xrcs_sensitivity(pulse=9391, write=False):
    """
    Test optimisation methods and use of different Te measurements but identical profile shapes
    on final results (Te and Ti + Wth)

    Test for 9539 saved to 25009539 runs 40, 41, 42, 43
    """

    # W-line emission moments, n3w PPAC result
    run_name = "RUN40"
    descr = f"Moment analysis w-line, Te(n3w)"
    res = plasma_workflow(
        pulse=pulse,
        tstart=0.02,
        tend=0.12,
        diagn_ne="smmh1",
        quant_te="te_n3w",
        imp_conc=(0.03, 0.001, 0.01),
        marchuk=True,
        xrcs_time=True,
        use_ratios=False,
        calc_error=False,
    )
    pl, raw_data, data, bckc = res

    dt_xrcs = raw_data["xrcs"]["ti_w"].t[1] - raw_data["xrcs"]["ti_w"].t[0]
    _pl = deepcopy(pl)
    _bckc = deepcopy(bckc)

    pulse = pl.pulse + 25000000
    if write:
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name
        )

    # # W-line emission moments, kw PPAC result
    # run_name = "RUN41"
    # descr = f"Moment analysis w-line, Te(kw)"
    # pl = deepcopy(_pl)
    # bckc = deepcopy(_bckc)
    # bckc = pl.match_xrcs_temperatures(
    #     data,
    #     bckc=bckc,
    #     quantity_te="te_kw",
    #     use_ratios=False,
    #     use_satellites=False,
    #     calc_error=False,
    # )
    # pl, bckc = propagate(
    #     pl, data, bckc, quant_ar="int_w", cal_ar=0.03, dt=dt_xrcs,
    # )
    # if write:
    #     hda_tree.write(
    #         pl,
    #         pulse,
    #         "HDA",
    #         data=data,
    #         bckc=bckc,
    #         descr=descr,
    #         run_name=run_name,
    #     )
    #
    # # k/w ratio
    # run_name = "RUN42"
    # descr = f"Line ratios, Te(kw)"
    # pl = deepcopy(_pl)
    # bckc = deepcopy(_bckc)
    # bckc = pl.match_xrcs_temperatures(
    #     data,
    #     bckc=bckc,
    #     quantity_te="te_kw",
    #     use_ratios=True,
    #     use_satellites=False,
    #     calc_error=False,
    # )
    # pl, bckc = propagate(
    #     pl, data, bckc, quant_ar="int_w", cal_ar=0.03, dt=dt_xrcs,
    # )
    # if write:
    #     hda_tree.write(
    #         pl,
    #         pulse,
    #         "HDA",
    #         data=data,
    #         bckc=bckc,
    #         descr=descr,
    #         run_name=run_name,
    #     )

    # n3/w ratio
    run_name = "RUN43"
    descr = f"Line ratios, Te(n3w)"
    pl = deepcopy(_pl)
    bckc = deepcopy(_bckc)
    bckc = pl.match_xrcs_temperatures(
        data, bckc=bckc, quantity_te="te_n3w", use_ratios=True, calc_error=False,
    )
    pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
    if write:
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
        )

    # n3/w ratio more peaked Te profile
    run_name = "RUN44"
    descr = f"Line ratios, Te(n3w), more peaked Te"
    pl = deepcopy(_pl)
    bckc = deepcopy(_bckc)
    bckc = pl.match_xrcs_temperatures(
        data,
        bckc=bckc,
        quantity_te="te_n3w",
        use_ratios=True,
        calc_error=False,
        wped=1.5,
    )
    pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
    if write:
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
        )

    # n3/w ratio more peaked Ne profiles (Nimp following Ne)
    run_name = "RUN45"
    descr = f"Line ratios, Te(n3w), more peaked Ne"
    pl = deepcopy(_pl)
    bckc = deepcopy(_bckc)
    pl.Ne_prof.wped = 4
    pl.Ne_prof.build_profile()
    pl.Nimp_prof.wped = 4
    pl.Nimp_prof.build_profile()
    bckc = pl.match_interferometer(data, bckc=bckc, diagnostic="smmh1", quantity="ne")
    pl.calc_imp_dens()
    bckc = pl.match_xrcs_temperatures(
        data, bckc=bckc, quantity_te="te_n3w", use_ratios=True, calc_error=False,
    )
    pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
    if write:
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
        )

    # n3/w ratio more peaked Ne profiles (Nimp following Ne) and Te
    run_name = "RUN46"
    descr = f"Line ratios, Te(n3w), more peaked Te, more peaked Ne"
    pl = deepcopy(_pl)
    bckc = deepcopy(_bckc)
    pl.Ne_prof.wped = 5
    pl.Ne_prof.build_profile()
    pl.Nimp_prof.wped = 5
    pl.Nimp_prof.build_profile()
    bckc = pl.match_interferometer(data, bckc=bckc, diagnostic="smmh1", quantity="ne")
    pl.calc_imp_dens()
    bckc = pl.match_xrcs_temperatures(
        data,
        bckc=bckc,
        quantity_te="te_kw",
        use_ratios=True,
        calc_error=False,
        wped=1.5,
    )
    pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
    if write:
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
        )

    # n3/w ratio EVEN more peaked Ne profiles (Nimp following Ne) and Te
    run_name = "RUN47"
    descr = f"Line ratios, Te(n3w), more peaked Te, much more peaked Ne"
    pl = deepcopy(_pl)
    bckc = deepcopy(_bckc)
    pl.Ne_prof.wped = 5
    pl.Ne_prof.peaking = 1.8
    pl.Ne_prof.build_profile()
    pl.Nimp_prof.wped = 5
    pl.Ne_prof.peaking = 1.8
    pl.Nimp_prof.build_profile()
    bckc = pl.match_interferometer(data, bckc=bckc, diagnostic="smmh1", quantity="ne")
    pl.calc_imp_dens()
    bckc = pl.match_xrcs_temperatures(
        data,
        bckc=bckc,
        quantity_te="te_kw",
        use_ratios=True,
        calc_error=False,
        wped=1.5,
    )
    pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
    if write:
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
        )

    # n3/w ratio broad Ti
    run_name = "RUN48"
    descr = f"Line ratios, Te(n3w), more peaked Te, much more peaked Ne"
    pl = deepcopy(_pl)
    bckc = deepcopy(_bckc)
    bckc = pl.match_interferometer(data, bckc=bckc, diagnostic="smmh1", quantity="ne")
    pl.calc_imp_dens()
    bckc = pl.match_xrcs_temperatures(
        data,
        bckc=bckc,
        quantity_te="te_kw",
        use_ratios=True,
        calc_error=False,
        wped=1.5,
        use_ref=False,
    )
    pl, bckc = propagate(pl, raw_data, data, bckc, quant_ar="int_w")
    if write:
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name,
        )

    return pl, raw_data, data, bckc


def sawtoothing(
    pulse=9229,  #
    tstart=0.072,
    tend=0.078,
    dt=0.001,
    t_pre=0.073,
    t_post=0.075,
    r_inv=0.1,
    diagn_ne="smmh1",
    diagn_te="xrcs",
    quant_ne="ne",
    quant_te="te_n3w",
    quant_ti="ti_w",
    quant_ar="int_w",
    main_ion="h",
    impurities=("c", "ar", "he"),
    imp_conc=(0.03, 0.001, 0.01),
    cal_ar=1.0,  # 1.0e13
    marchuk=True,
    savefig=False,
    name="",
    pl=None,
    raw_data=None,
    data=None,
    bckc=None,
):
    """
    tests.sawtoothing()
    tests.sawtoothing(t_pre=0.081, t_post=0.0825, tstart=0.078, tend=0.086)
    tests.sawtoothing(pulse=9391, t_pre=0.0715, t_post=0.0815, tstart=0.07, tend=0.09)

    pulse=9229
    tstart=0.072
    tend=0.078
    dt=0.001
    t_pre=0.073
    t_post=0.075
    r_inv=0.1
    diagn_ne="smmh1"
    diagn_te="xrcs"
    quant_ne="ne"
    quant_te="te_n3w"
    quant_ti="ti_w"
    quant_ar="int_w"
    main_ion="h"
    impurities=("c", "ar", "he")
    imp_conc=(0.03, 0.001, 0.01)
    marchuk = True
    cal_ar=1.
    calc_error=False
    """

    if pl is None:
        raw = ST40data(pulse, tstart - 0.01, tend + 0.01)
        raw.get_all()
        raw_data = raw.data

        bckc = {}
        elements = list(main_ion)
        elements.extend(list(impurities))

        pl = Plasma(tstart=tstart, tend=tend, dt=dt, elements=elements)
        data = pl.build_data(raw_data, pulse=pulse)

        for i, elem in enumerate(impurities):
            if elem in pl.ion_conc.element:
                pl.ion_conc.loc[dict(element=elem)] = imp_conc[i]

        # Find Ar density sawtooth crash, assuming Te stays constant
        if "xrcs" in raw_data:
            pl.forward_models["xrcs"] = XRCSpectrometer(marchuk=marchuk)
        if "princeton" in raw_data:
            pl.forward_models["princeton"] = PISpectrometer()

        pl.set_neutral_density(y1=1.0e15, y0=1.0e9)
        pl.build_atomic_data()
        pl.calculate_geometry()

    dt_xrcs = raw_data["xrcs"]["ti_w"].t[1] - raw_data["xrcs"]["ti_w"].t[0]

    # Reference times before and after the crash, inversion radius in rho
    t_pre = pl.time.values[np.argmin(np.abs(pl.time - t_pre).values)]
    t_post = pl.time.values[np.argmin(np.abs(pl.time - t_post).values)]
    t_mid = pl.t[np.abs(pl.t - (t_pre + t_post) / 2.0).argmin()]
    R_inv = pl.equilibrium.rmag.sel(t=t_post, method="nearest")
    z_inv = pl.equilibrium.zmag.sel(t=t_post, method="nearest") + r_inv
    rho_inv, _, _ = pl.equilibrium.flux_coords(R_inv, z_inv, t=t_post)

    # Find electron density sawtooth crash
    ne_pre_data = data["smmh1"]["ne"].sel(t=t_pre).values
    ne_post_data = data["smmh1"]["ne"].sel(t=t_post).values

    # Test different peaking factors to match both pre and post crash profiles
    pl.Ne_prof.y1 = 0.5e19
    pl.Ne_prof.wcenter = rho_inv.values / 2.0
    pl.Ne_prof.wped = 5
    pl.Ne_prof.peaking = 1.0
    pl.Ne_prof.build_profile()
    bckc = pl.match_interferometer(
        data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne,
    )
    pl.calc_imp_dens()
    volume = pl.volume.sel(t=t_post)

    scan = np.linspace(1.0, 2.5, 21)
    Ne_pre, Ne_post = [], []
    ne_pre_bckc, ne_post_bckc = [], []
    for s in scan:
        pre = deepcopy(pl.Ne_prof)
        pre.peaking = s
        pre.build_profile()
        pl.el_dens.loc[dict(t=t_pre)] = pre.yspl.values
        pre.y0 *= (
            ne_pre_data / pl.calc_ne_los_int(data[diagn_ne][quant_ne], t=t_pre)
        ).values
        pre.build_profile()
        pl.el_dens.loc[dict(t=t_pre)] = pre.yspl.values
        ne_pre_bckc.append(pl.calc_ne_los_int(data[diagn_ne][quant_ne], t=t_pre).values)
        Ne_pre.append(deepcopy(pre.yspl.values))
        pre_crash = pl.el_dens.sel(t=t_pre)
        post_crash = ph.sawtooth_crash(
            pre_crash.rho_poloidal, pre_crash.values, volume, rho_inv
        )
        Ne_post.append(deepcopy(post_crash))
        pl.el_dens.loc[dict(t=t_post)] = post_crash
        ne_post_bckc.append(
            pl.calc_ne_los_int(data[diagn_ne][quant_ne], t=t_post).values
        )

    ne_post_bckc = np.array(ne_post_bckc)

    plt.figure()
    raw_data["smmh1"]["ne"].plot()
    data["smmh1"]["ne"].plot(linewidth=3)
    ylim = plt.ylim()
    plt.vlines(t_post, ylim[0], ylim[1], color="black", linestyle="dashed")
    plt.plot(t_pre, ne_pre_data, marker="o", color="black")
    plt.plot(t_post, ne_post_data, marker="o", color="red")

    colors = cm.rainbow(np.linspace(0, 1, len(scan)))
    for i, s, in enumerate(scan):
        plt.plot(t_post, ne_post_bckc[i], "x", color=colors[i])

    if savefig:
        figname = plots.get_figname(pulse=pl.pulse, name=name)
        plots.save_figure(fig_name=f"{figname}data_electron_density_peaking_scan")

    ind = np.argmin(np.abs(ne_post_data - ne_post_bckc))
    plt.figure()
    for i, s, in enumerate(scan):
        plt.plot(pl.rho, Ne_pre[i], color=colors[i], alpha=0.5)
        plt.plot(pl.rho, Ne_post[i], color=colors[i], linestyle="dashed", alpha=0.5)

    plt.plot(pl.rho, Ne_pre[ind], color="black", marker="o")
    plt.plot(pl.rho, Ne_post[ind], color=colors[ind], linestyle="dashed", marker="o")

    if savefig:
        figname = plots.get_figname(pulse=pl.pulse, name=name)
        plots.save_figure(fig_name=f"{figname}profiles_electron_density_peaking_scan")

    # Fix electron density crash to best matching
    Ne_pre = DataArray(Ne_pre[ind], coords=[("rho_poloidal", pl.rho)])
    Ne_post = DataArray(Ne_post[ind], coords=[("rho_poloidal", pl.rho)])
    pl.el_dens = xr.where(pl.el_dens.t <= t_mid, Ne_pre, Ne_post)

    pl.calc_imp_dens()

    # Build tempertaure profiles to match XRCS using standard shapes
    bckc = pl.match_xrcs_temperatures(
        data,
        bckc=bckc,
        diagnostic=diagn_te,
        quantity_te=quant_te,
        quantity_ti=quant_ti,
        use_ratios=True,
        time=[t_pre, t_post],
    )
    t_xrcs = raw_data["xrcs"]["int_w"].t
    t_pre_xrcs = np.nanmax(xr.where(t_xrcs < t_pre, t_xrcs, np.nan).values)
    t_post_xrcs = np.nanmin(xr.where(t_xrcs > t_post, t_xrcs, np.nan).values)
    t_pre_xrcs = pl.time.values[np.argmin(np.abs(pl.time - t_pre_xrcs).values)]
    t_post_xrcs = pl.time.values[np.argmin(np.abs(pl.time - t_post_xrcs).values)]
    pl.el_temp = xr.where(
        pl.el_temp.t <= t_mid, pl.el_temp.sel(t=t_pre), pl.el_temp.sel(t=t_post)
    )
    for elem in pl.elements:
        ion_temp = pl.ion_temp.sel(element=elem)
        ion_temp = xr.where(
            ion_temp.t <= t_mid, ion_temp.sel(t=t_pre), ion_temp.sel(t=t_post)
        )
        pl.ion_temp.loc[dict(element=elem)] = ion_temp.values

    int_pre_data = data["xrcs"]["int_w"].sel(t=t_pre_xrcs, method="nearest").values
    int_post_data = data["xrcs"]["int_w"].sel(t=t_post_xrcs, method="nearest").values

    attrs = ["y0", "y1", "yend", "wped", "wcenter"]
    for a in attrs:
        setattr(pl.Nimp_prof, a, getattr(pl.Ne_prof, a))

    wcentre = 0.3
    wped = 1
    pl.Nimp_prof.peaking = 1.0
    pl.Nimp_prof.y1 = pl.Nimp_prof.y0 / 2.0
    pl.Nimp_prof.yend = pl.Nimp_prof.y1
    pl.Nimp_prof.wcenter = wcentre
    pl.Nimp_prof.wped = wped
    pl.Nimp_prof.build_profile()
    Nimp_first = deepcopy(pl.Nimp_prof)
    volume = pl.volume.sel(t=t_post)

    pl.calc_fz_lz()
    pl.calc_meanz()
    pl.calc_imp_dens()

    Nimp_pre, Nimp_post = [], []
    int_pre_bckc, int_post_bckc = [], []
    scan = np.linspace(1.0, 5, 21) ** 2
    for s in scan:
        # Pre crash profile
        pl.Nimp_prof = deepcopy(Nimp_first)
        pl.Nimp_prof.peaking = s
        pl.Nimp_prof.y1 /= s
        pl.Nimp_prof.yend /= s
        pl.Nimp_prof.wcenter = 0.12
        pl.Nimp_prof.build_profile()
        pl.ion_dens.loc[dict(element="ar", t=t_pre)] = pl.Nimp_prof.yspl.values
        bckc = pl.match_xrcs_intensity(
            data,
            bckc=bckc,
            diagnostic="xrcs",
            quantity=quant_ar,
            time=[t_pre],
            cal=cal_ar,
            dt=dt_xrcs,
        )
        int_pre_bckc.append(deepcopy(bckc["xrcs"][quant_ar].sel(t=t_pre).values))
        pre_crash = pl.ion_dens.sel(element="ar", t=t_pre)
        Nimp_pre.append(deepcopy(pre_crash.values))
        post_crash = ph.sawtooth_crash(
            pre_crash.rho_poloidal, pre_crash.values, volume, rho_inv
        )
        Nimp_post.append(deepcopy(post_crash))
        pl.ion_dens.loc[dict(element="ar", t=t_post)] = post_crash
        bckc = pl.match_xrcs_intensity(
            data,
            bckc=bckc,
            diagnostic="xrcs",
            quantity=quant_ar,
            time=[t_post],
            scale=False,
            cal=cal_ar,
            dt=dt_xrcs,
        )
        int_post_bckc.append(deepcopy(bckc["xrcs"][quant_ar].sel(t=t_post).values))

    int_pre_bckc = np.array(int_pre_bckc)
    int_post_bckc = np.array(int_post_bckc)

    plt.figure()
    raw_data["xrcs"]["int_w"].plot()
    data["xrcs"]["int_w"].plot(linewidth=3)
    ylim = plt.ylim()
    plt.vlines(t_post_xrcs, ylim[0], ylim[1], color="black", linestyle="dashed")
    plt.plot(t_pre_xrcs, int_pre_data, marker="o", color="black")
    plt.plot(t_post_xrcs, int_post_data, marker="o", color="red")

    colors = cm.rainbow(np.linspace(0, 1, len(scan)))
    for i, s, in enumerate(scan):
        plt.plot(t_post_xrcs, int_pre_bckc[i], "d", color=colors[i])
        plt.plot(t_post_xrcs, int_post_bckc[i], "x", color=colors[i])

    if savefig:
        figname = plots.get_figname(pulse=pl.pulse, name=name)
        plots.save_figure(fig_name=f"{figname}data_XRCS_argon_density_peaking_scan")

    plt.figure()
    ind = np.argmin(np.abs(int_post_data - int_post_bckc))
    for i, s, in enumerate(scan):
        plt.plot(pl.rho, Nimp_pre[i], color=colors[i], alpha=0.5)
        plt.plot(pl.rho, Nimp_post[i], color=colors[i], linestyle="dashed", alpha=0.5)

    plt.plot(pl.rho, Nimp_pre[ind], color="black", marker="o")
    plt.plot(pl.rho, Nimp_post[ind], color=colors[ind], linestyle="dashed", marker="o")

    if savefig:
        figname = plots.get_figname(pulse=pl.pulse, name=name)
        plots.save_figure(fig_name=f"{figname}profiles_XRCS_argon_density_peaking_scan")

    # Fix electron density crash to best matching
    Nimp_pre = DataArray(Nimp_pre[ind], coords=[("rho_poloidal", pl.rho)])
    Nimp_post = DataArray(Nimp_post[ind], coords=[("rho_poloidal", pl.rho)])
    Nimp = pl.ion_dens.sel(element="ar")
    Nimp = xr.where(Nimp.t <= t_mid, Nimp_pre, Nimp_post)
    pl.ion_dens.loc[dict(element="ar")] = Nimp.values

    # Build tempertaure profiles to match XRCS using standard shapes
    bckc = pl.match_xrcs_temperatures(
        data,
        bckc=bckc,
        diagnostic=diagn_te,
        quantity_te=quant_te,
        quantity_ti=quant_ti,
        use_ratios=True,
    )
    pl.calc_fz_lz()
    pl.calc_meanz()
    bckc = pl.match_xrcs_intensity(
        data,
        bckc=bckc,
        diagnostic="xrcs",
        quantity=quant_ar,
        scale=False,
        cal=cal_ar,
        dt=dt_xrcs,
    )
    pl.calc_main_ion_dens()
    pl.calc_zeff()
    pl.calc_rad_power()

    bckc = pl.calc_pressure()
    bckc = pl.interferometer(data, bckc=bckc)
    bckc = pl.bremsstrahlung(data, bckc=bckc)

    plots.compare_data_bckc(
        data, bckc, raw_data=raw_data, pulse=pl.pulse, savefig=savefig, name=name,
    )
    plots.profiles(pl, bckc=bckc, savefig=savefig, name=name)
    plots.time_evol(pl, data, bckc=bckc, savefig=savefig, name=name)

    return pl, raw_data, data, bckc


def compare_astra(pulse=8574, tstart=0.02, tend=0.14, revision=105, interf="nirh1"):
    pulse = 8574
    tstart = 0.02
    tend = 0.14
    revision = 105
    reader = ST40Reader(int(pulse + 25.0e6), tstart, tend, tree="astra")
    astra = reader.get("", "astra", revision)
    astra["ne"] *= 1.0e19
    astra["ni"] *= 1.0e19
    astra["te"] *= 1.0e3
    astra["ti"] *= 1.0e3
    # rho_poloidal = astra["p"].rho_poloidal
    # rho_toroidal = astra["ne"].rho_toroidal

    time = astra["te"].t
    tstart, tend = time[0], time[-1]
    dt = time[1] - time[0]
    pl, raw_data, data, bckc = plasma_workflow(
        pulse=pulse, tstart=tstart, tend=tend, dt=dt
    )


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
