from copy import deepcopy
import os
import pickle

from hda.models.spectrometer import XRCSpectrometer
import hda.hda_tree as hda_tree
from hda.plasma import initialize_bckc
from hda.plasma_new import Plasma, build_data, apply_limits

import hda.plots as plots
import hda.profiles as profiles
from hda.read_st40 import ST40data
import matplotlib.pylab as plt
import numpy as np
from scipy import constants
import xarray as xr

from indica.converters.time import bin_in_time_dt
from indica.operators.atomic_data import PowerLoss
from indica.readers import ADASReader
from indica.readers import ST40Reader

plt.ion()

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
    write=False,
    modelling=True,
    ne_peaking=None,
    marchuk=True,
    extrapolate=None,
    run_name="RUN40",
    descr="New profile shapes and ionisation balance",
    xrcs_time=True,
    use_ratios=True,
    calc_error=False,
    sxr=False,
    efit_pulse=None,
    efit_run=0,
):
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

    pl = Plasma(tstart=tstart, tend=tend, dt=dt, main_ion=main_ion, impurities=impurities, imp_conc=imp_conc)
    data = build_data(pl, raw_data, pulse=pulse)

    if ne_peaking is not None:
        pl.Ne_prof.peaking = ne_peaking
        pl.Ne_prof.build_profile()

    # Impose impurity concentration and calculate dilution
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
    bckc = pl.match_xrcs_intensity(
        data,
        bckc=bckc,
        diagnostic="xrcs",
        quantity=quant_ar,
    )
    bckc = pl.interferometer(data, bckc=bckc)
    bckc = pl.bremsstrahlung(data, bckc=bckc)

    if write:
        if modelling:
            pulse = pl.pulse + 25000000
        else:
            pulse = pl.pulse
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name
        )

    save_to_pickle(pl, raw_data, data, bckc, pulse=pl.pulse, name=run_name)

    return pl, raw_data, data, bckc

def run_all_scans(efit_pulse=None, efit_run=None, run_add="", force=True, calc_error=False):
    # pulses = [8532, 8533, 8605, 8621, 8875, 9098, 9099, 9229, 9401, 9486, 9537, 9538, 9539, 9619, 9622,
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

    pulses = [10009]
    tlims = [(0.02, 0.11)] * len(pulses)
    run_add = ["PROP"]*len(pulses)
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
            quant_te="te_kw",
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
            calc_error=calc_error,
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
    calc_error=False,
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
            calc_error=calc_error,
            sxr=sxr,
            main_ion=main_ion,
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
                        calc_error=calc_error,
                        use_ref=use_ref,
                    )
                    # propagate(pl, raw_data, data, bckc, quant_ar="int_w")

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

    return pl_dict, raw_data, data, bckc_dict, run_dict


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
    minmax=False,
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

        return pl_dict, raw_data, data, bckc_dict, astra_dict

    pulse = pl_dict["RUN60"].pulse

    # Selection of good runs
    # Stored energy inside uncertainty band of EFIT
    good_dict, chi2_dict = compare_runs(
        astra_dict, data["efit"]["wp"], perc_err=perc_err, key="wth"
    )

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
    print(f"Best run {best_run}")
    return pl, bckc


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

