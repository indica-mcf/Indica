"""
...write documentations...
"""

from copy import deepcopy
import pickle
import os

import hda.fac_profiles as fac
from hda.forward_models import Spectrometer
import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from xarray import DataArray

from trends.trends_database import Database
from indica.readers import ADASReader

# First pulse after Boronisation / GDC
BORONISATION = [8441, 8537, 9903]
GDC = [8545]
GDC = np.array(GDC) - 0.5

plt.ion()

def set_paths(database:Database, path_fig="", name_fig="",):
    if len(path_fig) == 0:
        path_fig = f"{os.path.expanduser('~')}/figures/regr_trends/"
    if len(name_fig) == 0:
        name_fig = f"{database.pulse_start}_{database.pulse_end}"

    return path_fig, name_fig

def plot_time_evol(
    database,
    info,
    to_plot,
    pulse_lim=None,
    tplot=None,
    savefig=False,
    vlines=True,
    path_fig="",
    name_fig="",
):
    if savefig:
        plt.ioff()
    if tplot is None:
        tit_front = "Maximum of "
        tit_end = ""
        result = database.max_val
    else:
        tit_front = ""
        tit_end = f" @ t={tplot:.3f} s"
        result = database.binned
    if pulse_lim is None:
        pulse_lim = (np.min(database.pulses) - 5, np.max(database.pulses) + 5)

    ymin = []
    ymax = []
    for title, ykey in to_plot.items():
        name = "time_evol"
        plt.figure()
        for i, k in enumerate(ykey):
            name += f"_{k}"
            res = result[k] * info[k]["const"]
            if tplot is not None:
                res = res.sel(t=tplot, method="nearest")

            x = res.pulse
            yval = flat(res.value)
            ymin.append(np.nanmin(yval))
            ymax.append(np.nanmax(yval))
            yerr = flat(res.error)

            if len(ykey) > 1:
                label = info[k]["label"]
            else:
                label = ""

            plt.errorbar(
                x, yval, yerr=yerr, fmt="o", label=label, alpha=0.5,
            )
        plt.xlabel("Pulse #")
        plt.ylabel(info[k]["units"])
        plt.title(f"{tit_front}{title}{tit_end}")
        plt.xlim(pulse_lim)

        if "sign" in info[k].keys():
            sign = info[k]["sign"]
        else:
            sign = +1

        if sign > 0:
            # plt.ylim(bottom=0, top=np.nanmax(ymax)*1.2)
            plt.ylim(bottom=0)
        elif sign < 0:
            # plt.ylim(bottom=np.nanmin(ymin)*0.8, top=0)
            plt.ylim(top=0)
        else:
            ylim = plt.ylim()

        if len(ykey) > 1:
            plt.legend()
        if vlines:
            add_vlines(BORONISATION)
            add_vlines(GDC, color="r")

        if tplot is None:
            name += "_max"
        else:
            name += f"_t_{tplot:.3f}s"

        if savefig:
            save_figure(path_fig, f"{name_fig}_{name}")
    if savefig:
        plt.ion()

    plt.figure()
    cols = cm.rainbow(np.linspace(0, 1, len(database.pulses)))
    for i, p in enumerate(database.pulses):
        (
            database.binned["ipla_efit"].value.sel(pulse=p)
            * database.info["ipla_efit"]["const"]
        ).plot(color=cols[i], alpha=0.5)

    plt.title(f"Pulse range [{np.min(database.pulses)}, {np.max(database.pulses)}]")
    plt.xlabel("time (s)")
    plt.ylabel(
        f"{database.info['ipla_efit']['label']} {database.info['ipla_efit']['units']}"
    )
    plt.ylim(bottom=0)

    plt.figure()
    bvl_ov_ip = database.binned["i_bvl"] / database.binned["ipla_efit"]
    cols = cm.rainbow(np.linspace(0, 1, len(database.pulses)))
    for i, p in enumerate(bvl_ov_ip.pulse):
        bvl_ov_ip.value.sel(pulse=p).plot(color=cols[i], alpha=0.5)

    plt.title(f"Pulse range [{np.min(database.pulses)}, {np.max(database.pulses)}]")
    plt.xlabel("time (s)")
    plt.ylabel("I$_{BVL}$/I$_P$")
    plt.ylim(top=0)

    plt.figure()
    time = np.arange(0.01, 0.2, 0.02)
    cols = cm.rainbow(np.linspace(0, 1, len(time)))
    for i, t in enumerate(time):
        bvl_ov_ip.value.sel(t=t, method="nearest").plot(
            color=cols[i], label=f"{t:.2f} s", alpha=0.5, marker="o"
        )

    plt.legend()
    plt.title(f"Time range [{time.min():.2f}, {time.max():.2f}]")
    plt.xlabel("Pulse")
    plt.ylabel("I$_{BVL}$/I$_P$")
    plt.ylim(top=0)


def plot_bivariate(
    filtered, info, to_plot, label=None, savefig=False, path_fig="", name_fig="",
):

    if savefig:
        plt.ioff()

    for title, keys in to_plot.items():
        xkey, ykey = keys
        name = f"{ykey}_vs_{xkey}"
        xinfo = info[xkey]
        yinfo = info[ykey]

        plt.figure()
        for label, data in filtered.items():
            binned = data["binned"]
            x = binned[xkey] * xinfo["const"]
            y = binned[ykey] * yinfo["const"]
            xval = flat(x.value)
            xerr = flat(x.error)
            yval = flat(y.value)
            yerr = flat(y.error)

            plt.errorbar(
                xval, yval, xerr=xerr, yerr=yerr, fmt="o", label=label, alpha=0.5,
            )
            plt.xlabel(xinfo["label"] + " " + xinfo["units"])
            plt.ylabel(yinfo["label"] + " " + yinfo["units"])
            plt.title(title)
        if label is not None:
            plt.legend()

        if savefig:
            save_figure(path_fig, f"{name_fig}_{name}")
    if savefig:
        plt.ion()


def plot_trivariate(
    filtered, info, to_plot, nbins=10, savefig=False, path_fig="", name_fig=""
):

    if savefig:
        plt.ioff()

    for title, keys in to_plot.items():
        xkey, ykey, zkey = keys
        xinfo = info[xkey]
        yinfo = info[ykey]
        zinfo = info[zkey]

        ylim = []
        xlim = []
        for i, label in enumerate(filtered.keys()):
            binned = filtered[label]["binned"]
            x = binned[xkey].value.values.flatten() * xinfo["const"]
            y = binned[ykey].value.values.flatten() * yinfo["const"]
            xlim.append([np.nanmin(x), np.nanmax(x)])
            ylim.append([np.nanmin(y), np.nanmax(y)])
        xlim = np.array(xlim)
        ylim = np.array(ylim)
        xlim = (np.min(xlim) * 0.95, np.max(xlim) * 1.05)
        ylim = (np.min(ylim) * 0.95, np.max(ylim) * 1.05)
        for i, label in enumerate(filtered.keys()):
            plt.figure()
            name = f"{zkey}_vs_{ykey}_and_{xkey}"
            binned = filtered[label]["binned"]
            x = binned[xkey].value.values.flatten() * xinfo["const"]
            y = binned[ykey].value.values.flatten() * yinfo["const"]
            z = binned[zkey].value.values.flatten() * zinfo["const"]
            xerr = binned[xkey].error.values.flatten() * xinfo["const"]
            yerr = binned[ykey].error.values.flatten() * yinfo["const"]

            zhist = np.histogram(z[np.where(np.isfinite(z))[0]], bins=nbins)
            bins = zhist[1]
            bins_str = [f"{b:.1f}" for b in bins]
            nbins = len(bins) - 1

            ind = np.argsort(z)
            cmap = plt.cm.get_cmap("rainbow", nbins)
            sc = plt.scatter(
                x[ind],
                y[ind],
                c=z[ind],
                s=40,
                vmin=np.nanmin(z),
                vmax=np.nanmax(z),
                cmap=cmap,
                marker="o",
                alpha=0.5,
            )
            plt.colorbar(sc, label=f"{zinfo['label']} {zinfo['units']}")

            cols = cm.rainbow(np.linspace(0, 1, nbins))
            for ib in range(nbins):
                ind = np.where((z >= bins[ib]) * (z < bins[ib + 1]))
                plt.errorbar(
                    x[ind],
                    y[ind],
                    xerr=xerr[ind],
                    yerr=yerr[ind],
                    fmt="o",
                    color=cols[ib],
                    label=f"[{bins_str[ib]}, {bins_str[ib+1]}]",
                    alpha=0.2,
                )

            plt.xlabel(xinfo["label"] + " " + xinfo["units"])
            plt.ylabel(yinfo["label"] + " " + yinfo["units"])
            plt.title(f"{label}")
            plt.xlim(xlim)
            plt.ylim(ylim)
            name += f"_{label}"
            if savefig:
                save_figure(path_fig, f"{name_fig}_{name}")
    if savefig:
        plt.ion()


def plot_hist(
    filtered,
    info,
    to_plot,
    tplot=None,
    bins=None,
    savefig=False,
    path_fig="",
    name_fig="",
):
    if savefig:
        plt.ioff()

    for title, key in to_plot.items():
        res = []
        labels = []
        for label, data in filtered.items():
            res_tmp = data["binned"][key] * info[key]["const"]
            if tplot is not None:
                res_tmp = res_tmp.sel(t=tplot, method="nearest")
            res_tmp = flat(res_tmp.value)
            res.append(res_tmp)
            labels.append(label)

        plt.figure()
        name = f"hist_{key}"
        plt.hist(res, bins=bins, label=labels, density=True)
        plt.title(f"{info[key]['label']}")
        plt.xlabel(info[key]["units"])
        plt.legend()
        if tplot is not None:
            name += f"_t_{tplot:.3f}s"
        if savefig:
            save_figure(path_fig, f"{name_fig}_{name}")
    if savefig:
        plt.ion()


def max_ti_pulses(database, savefig=False, plot_results=False):
    cond_general = {
        "nbi_power": {"var": "value", "lim": (20, np.nan)},
        "te_xrcs": {"var": "error", "lim": (np.nan, 0.2)},
        "ti_xrcs": {"var": "error", "lim": (np.nan, 0.2)},
        "ipla_efit": {"var": "gradient", "lim": (-1.0e6, np.nan)},
    }
    cond_special = deepcopy(cond_general)
    cond_special["ti_xrcs"] = {"var": "value", "lim": (1.5e3, np.nan)}
    cond = {
        "NBI": cond_general,
        "NBI & Ti > 1.5 keV": cond_special,
    }
    filtered = apply_selection(database.binned, cond, default=False)
    if plot_results or savefig:
        plot(database, filtered, savefig=savefig)

    print("\n Pulses in selection")
    for k in filtered.keys():
        print(f"\n {k}", filtered[k]["pulses"])

    return filtered


def ip_400_500(database, savefig=False, plot_results=False):
    cond_general = {
        "nbi_power": {"var": "value", "lim": (20, np.nan)},
        "te_xrcs": {"var": "error", "lim": (np.nan, 0.2)},
        "ti_xrcs": {"var": "error", "lim": (np.nan, 0.2)},
        "ipla_efit": {"var": "gradient", "lim": (-1.0e6, np.nan)},
    }
    cond_special = deepcopy(cond_general)
    cond_special["ipla_efit"] = {"var": "value", "lim": (0.4e6, 0.5e6)}
    cond = {
        "NBI": cond_general,
        "NBI & Ip = [400, 500] kA": cond_special,
    }
    filtered = apply_selection(database.binned, cond, default=False)
    if plot_results or savefig:
        plot(database, filtered, savefig=savefig, plot_time=False)

    print("Pulses in selection")
    for k in filtered.keys():
        print(k, filtered[k]["pulses"])

    return filtered

def apply_selection(
    binned, cond=None, default=True,
):
    """
    Apply selection criteria as defined in the cond dictionary

    Parameters
    ----------
    binned
        Database class result dictionary of binned quantities
    cond
        Dictionary of selection criteria (see default defined below)
        Different elements in list give different selection, elements
        in sub-dictionary are applied together (&)
    default
        set selection criteria conditions to default defined below

    Returns
    -------

    """
    # TODO: max_val calculation too time-consuming...is it worth it?
    if default:
        cond = {
            "NBI": {"nbi_power": {"var": "value", "lim": (20, np.nan)},},
            "Ohmic": {"nbi_power": {"var": "value", "lim": (0,)},},
        }
    # "te0": {"var": "error", "lim": (np.nan, 0.2)},
    # "ti0": {"var": "error", "lim": (np.nan, 0.2)},

    # Apply selection criteria
    if cond is not None:
        filtered = deepcopy(cond)
        for kcond, c in cond.items():
            binned_tmp = deepcopy(binned)
            selection_tmp = selection_criteria(binned_tmp, c)
            for kbinned in binned_tmp.keys():
                binned_tmp[kbinned] = xr.where(
                    selection_tmp, binned_tmp[kbinned], np.nan
                )

            pulses = []
            for p in binned_tmp[kbinned].pulse:
                if any(selection_tmp.sel(pulse=p)):
                    pulses.append(p)
            pulses = np.array(pulses)

            filtered[kcond]["binned"] = binned_tmp
            filtered[kcond]["selection"] = selection_tmp
            filtered[kcond]["pulses"] = pulses
    else:
        filtered = {"All": {"selection": None, "binned": binned}}

    return filtered



def plot(
    database, filtered=None, tplot=0.03, savefig=False, plot_time=True, pulse_lim=None
):
    if savefig:
        plt.ioff()

    if filtered is None:
        if hasattr(database, "filtered"):
            filtered = database.filtered

    path_fig, name_fig = set_paths(database)

    info = database.info

    ###################################
    # Simulated XRCS measurements
    ###################################
    if plot_time == True:
        if hasattr(database, "temp_ratio"):
            plt.figure()
            temp_ratio = database.temp_ratio
            for i in range(len(temp_ratio)):
                plt.plot(temp_ratio[i].te0, temp_ratio[i].te_xrcs)

            plt.plot(temp_ratio[0].te0, temp_ratio[0].te0, "--k", label="Central Te")
            plt.legend()
            add_to_plot(
                "T$_e$(0)", "T$_{e,i}$(XRCS)", "XRCS measurement vs. Central Te",
            )
            if savefig:
                save_figure(path_fig, f"{name_fig}_XRCS_Te0_parametrization")

            plt.figure()
            for i in range(len(temp_ratio)):
                el_temp = temp_ratio[i].attrs["el_temp"]
                plt.plot(
                    el_temp.rho_poloidal,
                    el_temp.sel(t=el_temp.t.mean(), method="nearest") / 1.0e3,
                )

            plt.legend()
            add_to_plot(
                "rho_poloidal", "T$_e$ (keV)", "Temperature profiles",
            )
            if savefig:
                save_figure(path_fig, f"{name_fig}_XRCS_parametrization_temperatures")

            plt.figure()
            for i in range(len(temp_ratio)):
                el_dens = temp_ratio[i].attrs["el_dens"]
                plt.plot(
                    el_dens.rho_poloidal,
                    el_dens.sel(t=el_dens.t.mean(), method="nearest") / 1.0e3,
                )

            plt.legend()
            add_to_plot(
                "rho_poloidal", "n$_e$ (10$^{19}$)", "Density profiles",
            )
            if savefig:
                save_figure(path_fig, f"{name_fig}_XRCS_parametrization_densities")

        ###################################
        # Time evolution of maximum quantities
        ###################################
        to_plot = {
            "Plasma Current": ("ipla_efit",),
            "Pulse Length": ("pulse_length",),
            "Stored Energy": ("wp_efit",),
            "Electron Temperature (XRCS)": ("te_xrcs",),
            "Ion Temperature (XRCS)": ("ti_xrcs",),
            "Ion/Electron Temperature (XRCS)": ("ti_te_xrcs",),
            "Electron Density": ("ne_nirh1",),
            "Electron Pressure": ("ne_nirh1_te_xrcs",),
            "BVL current": ("i_bvl",),
            "MC Current": ("imc",),
            "Gas pressure": ("gas_press",),
            "Gas prefill": ("gas_prefill",),
            "Cumulative gas puff": ("gas_cumulative",),
            "Cumulative NBI power": ("total_nbi",),
            "Plasma current @ 15 ms / MC current ": ("rip_imc",),
        }
        # "Electron Temperature": ("te_xrcs", "te0",),
        # "Ion Temperature": ("ti_xrcs", "ti0",),
        plot_time_evol(
            database,
            info,
            to_plot,
            savefig=savefig,
            path_fig=path_fig,
            name_fig=name_fig,
            pulse_lim=pulse_lim,
        )

        ###################################
        # Time evolution of quantities at specified tplot
        ###################################
        to_plot = {
            "Bremsstrahlung PI": ("brems_pi",),
            "Bremsstrahlung MP": ("brems_mp",),
            "Plasma Current": ("ipla_efit",),
            "Gas pressure": ("gas_press",),
            "Total gas puff": ("gas_cumulative",),
            "Electron Density": ("ne_nirh1",),
            "BVL current": ("i_bvl",),
            "H visible emission": ("sum_h",),
            "He visible emission": ("sum_he",),
            "B visible emission": ("sum_b",),
            "C visible emission": ("sum_c",),
            "N visible emission": ("sum_n",),
            "O visible emission": ("sum_o",),
            "Ar visible emission": ("sum_ar",),
        }
        # "H-alpha": ("h_i_6563",),
        # "Helium": ("he_ii_4686",),
        # "Boron": ("b_ii_3451",),
        # "Oxygen": ("o_iv_3063",),
        # "Argon": ("ar_ii_4348",),

        plot_time_evol(
            database,
            info,
            to_plot,
            tplot=tplot,
            savefig=savefig,
            path_fig=path_fig,
            name_fig=name_fig,
            pulse_lim=pulse_lim,
        )

    ###################################
    # Bivariate distributions for data-points which satisfy selection criteria
    ###################################
    if filtered is not None:
        to_plot = {
            "T$_e$(XRCS) vs. I$_P$": ("ipla_efit", "te_xrcs"),
            "T$_i$(XRCS) vs. I$_P$": ("ipla_efit", "ti_xrcs"),
            "T$_i$(XRCS) vs. n$_e$(NIRH1)": ("ne_nirh1", "ti_xrcs"),
            "T$_i$(XRCS) vs. gas pressure": ("gas_press", "ti_xrcs"),
            "T$_i$(XRCS) vs. Cumulative gas puff": ("gas_cumulative", "ti_xrcs"),
            "T$_i$(XRCS) vs. Electron pressure": ("ne_nirh1_te_xrcs", "ti_xrcs"),
        }

        plot_bivariate(
            filtered,
            info,
            to_plot,
            savefig=savefig,
            path_fig=path_fig,
            name_fig=name_fig,
        )

        to_plot = {
            "Plasma Current": "ipla_efit",
            "Electron Density": "ne_nirh1",
            "XRCS Electron Temperature": "te_xrcs",
            "XRCS Ion Temperature": "ti_xrcs",
            "Gas Pressure": "gas_press",
            "Cumulative Gas Puff": "gas_cumulative",
        }

        plot_hist(
            filtered,
            info,
            to_plot,
            tplot=None,
            bins=None,
            savefig=savefig,
            path_fig=path_fig,
            name_fig=name_fig,
        )

        to_plot = {
            "Plasma Current": ("te_xrcs", "ti_xrcs", "ipla_efit"),
            "Electron Density": ("te_xrcs", "ti_xrcs", "ne_nirh1"),
            "Gas Pressure, Te, Ti": ("te_xrcs", "ti_xrcs", "gas_press"),
            "Gas Pressure, Ne, Ti": ("ne_nirh1", "ti_xrcs", "gas_press"),
            "Cumulative Gas Puff, Te, Ti": ("te_xrcs", "ti_xrcs", "gas_cumulative"),
            "Cumulative Gas Puff, Ne, Ti": ("ne_nirh1", "ti_xrcs", "gas_cumulative"),
        }

        # filtered["All"] = {"selection": None, "binned": database.binned}
        plot_trivariate(
            filtered,
            info,
            to_plot,
            savefig=savefig,
            path_fig=path_fig,
            name_fig=name_fig,
        )

    if savefig:
        plt.ion()


def add_vlines(xvalues, color="k"):
    ylim = plt.ylim()
    for b in xvalues:
        plt.vlines(b, ylim[0], ylim[1], linestyles="dashed", colors=color, alpha=0.5)


def save_figure(path_fig, name_fig, orientation="landscape", ext=".jpg"):
    plt.savefig(
        path_fig + name_fig + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )


def simulate_xrcs(pickle_file="XRCS_temperature_parametrization.pkl", write=False):
    print("Simulating XRCS measurement for Te(0) re-scaling")

    adasreader = ADASReader()
    xrcs = Spectrometer(
        adasreader, "ar", "16", transition="(1)1(1.0)-(1)0(0.0)", wavelength=4.0,
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


def calc_mean_std(time, data, tstart, tend, lower=0.0, upper=None, toffset=None):

    avrg = np.nan
    std = 0.0
    offset = 0
    if (
        not np.array_equal(data, "FAILED")
        and (np.size(data) == np.size(time))
        and np.size(data) > 1
    ):
        it = (time >= tstart) * (time <= tend)
        if lower is not None:
            it *= data > lower
        if upper is not None:
            it *= data < upper

        it = np.where(it)[0]
        if len(it) > 1:
            if toffset is not None:
                it_offset = np.where(time <= toffset)[0]
                if len(it_offset) > 1:
                    offset = np.mean(data[it_offset])

            avrg = np.mean(data[it] + offset)
            if len(it) >= 2:
                std = np.std(data[it] + offset)

    return avrg, std


def add_to_plot(xlab, ylab, tit, legend=True, vlines=False):
    if vlines:
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    if legend:
        plt.legend()


def flat(data: DataArray):
    return data.values.flatten()
