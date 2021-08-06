"""Read and plot time evolution of various quantities
at identical times in a discharge over a defined pulse range

Example call:

    import hda.analyse_trends as trends
    corr = trends.correlations(8400, 8534, t=[0.03, 0.08])


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

from indica.readers import ADASReader
from indica.readers import ST40Reader

# First pulse after Boronisation / GDC
BORONISATION = [8441, 8537]
GDC = [8545]
# for p in np.arange(8547, 8560+1):
#     GDC.append(p)
GDC = np.array(GDC) - 0.5

plt.ion()


class Database:
    def __init__(
        self,
        pulse_start=8207,
        pulse_end=8626,
        tlim=(-0.03, 0.3),
        dt=0.01,
        overlap=0.5,
        t_max=0.02,
        reload=False,
    ):

        self.info = get_data_info()
        if getpass.getuser() == "lavoro":
            self.path = "/Users/lavoro/Work/Python/Indica_github/data/"
        else:
            self.path = f"/home/{getpass.getuser()}/python/"
        self.filename = f"{pulse_start}_{pulse_end}_regression_database.pkl"
        if reload:
            regr_data = pickle.load(open(self.path + self.filename, "rb"))
            self.pulse_start = regr_data.pulse_start
            self.pulse_end = regr_data.pulse_end
            self.tlim = regr_data.tlim
            self.dt = regr_data.dt
            self.overlap = regr_data.overlap
            self.time = regr_data.time
            self.empty_binned = regr_data.empty_binned
            self.binned = regr_data.binned
            self.max_val = regr_data.max_val
            self.pulses = regr_data.pulses
            self.t_max = regr_data.t_max
        else:
            self.pulse_start = pulse_start
            self.pulse_end = pulse_end
            self.tlim = tlim
            self.dt = dt
            self.overlap = overlap
            self.time = np.arange(self.tlim[0], self.tlim[1], dt * overlap)
            self.t_max = t_max

            # Initialize data structure
            self.initialize_structures()

            # Read all data
            self.read_data(pulse_start, pulse_end)

    def __call__(self, *args, **kwargs):
        """
        Apply general filters to data
        """

        # Conversion factor to calculate central temperatures
        temp_ratio = simulate_xrcs()
        self.temp_ratio = temp_ratio

        # Apply general filters
        self.binned = general_filters(self.binned)

        # Estimate central temperature from parameterization
        self.binned = calc_central_temperature(self.binned, self.temp_ratio)

        # Calculate max values of binned data
        self.max_val = calc_max_val(
            self.binned, self.max_val, self.info, t_max=self.t_max
        )

        # Apply general filters
        self.max_val = general_filters(self.max_val)

        # Apply defult selection criteria
        self.filtered = apply_selection(self.binned)

    def initialize_structures(self):
        value = DataArray(np.full(self.time.shape, np.nan), coords=[("t", self.time)])
        error = xr.full_like(value, 0)
        gradient = xr.full_like(value, np.nan)
        cumul = xr.full_like(value, np.nan)
        self.empty_binned = Dataset(
            {"value": value, "error": error, "gradient": gradient, "cumul": cumul}
        )

        value = DataArray(0.0)
        error = xr.full_like(value, 0)
        time = xr.full_like(value, np.nan)
        self.empty_max_val = Dataset({"value": value, "error": error, "time": time})

    def read_data(self, pulse_start, pulse_end):
        """
        Read data in time-range of interest

        Parameters
        ----------
        pulse_start
            first pulse in range
        pulse_end
            last pulse in range

        Returns
        -------

        """

        binned = {}
        max_val = {}
        for k in self.info.keys():
            binned[k] = []
            max_val[k] = []

        pulses_all = []
        for pulse in np.arange(pulse_start, pulse_end + 1):
            print(pulse)
            reader = ST40Reader(
                int(pulse),
                self.tlim[0],
                self.tlim[1],
            )

            time, _ = reader._get_signal("", "pfit", ".post_best.results:time", -1)
            if np.array_equal(time, "FAILED"):
                print("no Ip from PFIT")
                continue
            if np.min(time) > np.max(self.tlim) or np.max(time) < np.min(self.tlim):
                print("no Ip from PFIT in time range")
                continue
            ipla, _ = reader._get_signal("", "pfit", ".post_best.results.global:ip", -1)
            if np.array_equal(time, "FAILED"):
                print("no Ip from PFIT")
                continue

            tind = np.where(ipla > 50.0e3)[0]
            if len(tind) < 3:
                print("max(Ip) from PFIT < 50 kA")
                continue

            tlim_pulse = (np.min([np.min(time[tind]), -0.01]), np.max(time[tind]))

            pulses_all.append(np.array([pulse] * len(self.time)))

            for k, v in self.info.items():
                if "uid" not in v.keys():
                    binned[k].append(self.empty_binned)
                    max_val[k].append(self.empty_max_val)
                    continue

                err = None
                data, dims = reader._get_data(v["uid"], v["diag"], v["node"], v["seq"])
                if np.array_equal(data, "FAILED"):
                    binned[k].append(self.empty_binned)
                    max_val[k].append(self.empty_max_val)
                    continue

                time = dims[0]
                if v["err"] is not None:
                    err, _ = reader._get_data(v["uid"], v["diag"], v["err"], v["seq"])

                if np.min(time) > np.max(self.tlim) or np.max(time) < np.min(self.tlim):
                    print(f"{k} wrong time range")
                    binned[k].append(self.empty_binned)
                    max_val[k].append(self.empty_max_val)
                    continue

                binned_tmp, max_val_tmp = self.bin_in_time(
                    data, time, err, tlim=tlim_pulse
                )

                binned[k].append(binned_tmp)
                max_val[k].append(max_val_tmp)

        self.pulses_all = np.array(pulses_all)
        self.pulses = np.unique(np.array(self.pulses_all).flatten())

        for k in binned.keys():
            binned[k] = xr.concat(binned[k], "pulse")
            binned[k] = binned[k].assign_coords({"pulse": self.pulses})
            max_val[k] = xr.concat(max_val[k], "pulse")
            max_val[k] = max_val[k].assign_coords({"pulse": self.pulses})

        binned["nbi_power"] = deepcopy(binned["hnbi"])

        self.binned = binned
        self.max_val = max_val

    def bin_in_time(self, data, time, err=None, tlim=(0, 0.5)):
        """
        Bin data in time, calculate maximum value in specified time range,
        create datasets for binned data and maximum values

        Parameters
        ----------
        data
            array of data to be binned
        time
            time axis of data to be binned
        err
            error of data to be binned
        tlim
            time limits to apply binnning and to search for maximum value

        Returns
        -------

        """
        time_new = self.time
        dt = self.dt
        binned = deepcopy(self.empty_binned)
        max_val = deepcopy(self.empty_max_val)

        ifin = np.where(np.isfinite(data))[0]
        if len(ifin) < 2:
            return binned, max_val

        time_binning = time_new[np.where((time_new >= tlim[0]) * (time_new < tlim[1]))]
        for t in time_binning:
            tind = (time >= t - dt / 2.0) * (time < t + dt / 2.0)
            tind_lt = time <= t

            if len(tind) > 0:
                data_tmp = data[tind]
                ifin = np.where(np.isfinite(data_tmp))[0]
                if len(ifin) >= 1:
                    binned.value.loc[dict(t=t)] = np.mean(data_tmp[ifin])
                    binned.cumul.loc[dict(t=t)] = np.sum(data[tind_lt]) * dt
                    binned.error.loc[dict(t=t)] = np.std(data_tmp[ifin])
                    if err is not None:
                        err_tmp = err[tind]
                        binned.error.loc[dict(t=t)] += np.sqrt(
                            np.sum(err_tmp[ifin] ** 2)
                        ) / len(ifin)

        binned.gradient.values = binned.value.differentiate("t")
        binned.cumul.values = xr.where(np.isfinite(binned.value), binned.cumul, np.nan)

        tind = np.where((time >= tlim[0]) * (time < tlim[1]))[0]
        max_ind = np.nanargmax(data[tind])
        max_val.value.values = data[tind[max_ind]]
        max_val.time.values = time[tind[max_ind]]
        if err is not None:
            max_val.error.values = err[tind[max_ind]]

        return binned, max_val

    # def add_data(self, pulse_end):
    #     """
    #     Add data from newer pulses to binned dictionary
    #
    #     Parameters
    #     ----------
    #     pulse_end
    #         Last pulse to include in the analysis
    #     """
    #     pulse_start = np.array(self.results["pulses"]).max() + 1
    #     if pulse_end < pulse_start:
    #         print("Newer pulses only (for the time being...)")
    #         return
    #     new = self.read_data(pulse_start, pulse_end)
    #
    #     for i, pulse in enumerate(new["pulses"]):
    #         for k1, res in new.items():
    #             if k1 == "pulses":
    #                 continue
    #             if type(res) != dict:
    #                 self.results[k1].append(res[i])
    #                 continue
    #
    #             for k2, res2 in res.items():
    #                 self.results[k1][k2].append(res2[i])


def general_filters(results):
    """
    Apply general filters to data read e.g. NBI power 0 where not positive
    """
    print("Applying general data filters")

    # Set all negative values to 0
    neg_to_zero = ["nbi_power"]
    for k in neg_to_zero:
        results[k] = xr.where(
            (results[k] > 0) * np.isfinite(results[k]),
            results[k],
            0,
        )

    # Set all negative values to Nan
    neg_to_nan = [
        "te_xrcs",
        "ti_xrcs",
        "ti0",
        "te0",
        "ipla_efit",
        "ipla_pfit",
        "wp_efit",
        "ne_nirh1",
        "ne_smmh1",
        "gas_press",
    ]
    for k in neg_to_nan:
        results[k] = xr.where(
            (results[k] > 0) * (np.isfinite(results[k])),
            results[k],
            np.nan,
        )

    # Set to Nan if values outside specific ranges
    err_perc_cond = {"var": "error", "lim": (np.nan, 0.2)}
    keys = [
        "te_xrcs",
        "ti_xrcs",
        "te0",
        "ti0",
        "brems_pi",
        "brems_mp",
        "h_i_6563",
        "he_ii_4686",
        "b_ii_3451",
        "o_iv_3063",
        "ar_ii_4348",
        "ne_smmh1",
        "wp_efit",
        "rip_pfit",
        "imc",
    ]

    for k in keys:
        cond = {k: err_perc_cond}
        selection = selection_criteria(results, cond)
        results[k] = xr.where(selection, results[k], np.nan)

    return results


def calc_central_temperature(binned, temp_ratio):
    print("Calculating central temperature from parameterization")

    # Central temperatures from XRCS parametrization
    mult_binned = []
    profs = np.arange(len(temp_ratio))
    for i in range(len(temp_ratio)):
        ratio_tmp = xr.full_like(binned["te_xrcs"].value, np.nan)
        # TODO: DataArray interp crashing if all nans (@ home only)
        for p in binned["te_xrcs"].pulse:
            te_xrcs = binned["te_xrcs"].value.sel(pulse=p)
            if any(np.isfinite(te_xrcs)):
                ratio_tmp.loc[dict(pulse=p)] = np.interp(
                    te_xrcs.values,
                    temp_ratio[i].te_xrcs,
                    temp_ratio[i].values,
                )
        mult_binned.append(ratio_tmp)
    mult_binned = xr.concat(mult_binned, "prof")
    mult_binned = mult_binned.assign_coords({"prof": profs})

    # Binned data
    mult_max = mult_binned.max("prof", skipna=True)
    mult_min = mult_binned.min("prof", skipna=True)
    mult_mean = mult_binned.mean("prof", skipna=True)
    binned["te0"].value.values = (binned["te_xrcs"].value * mult_mean).values
    err = np.abs(binned["te0"].value * mult_max - binned["te0"].value * mult_min)
    binned["te0"].error.values = np.sqrt(
        (binned["te_xrcs"].error * mult_mean) ** 2 + err ** 2
    ).values
    binned["ti0"].value.values = (binned["ti_xrcs"].value * mult_mean).values
    err = np.abs(binned["ti0"].value * mult_max - binned["ti0"].value * mult_min)
    binned["ti0"].error.values = np.sqrt(
        (binned["ti_xrcs"].error * mult_mean) ** 2 + err ** 2
    ).values

    return binned


def calc_max_val(binned, max_val, info, t_max=0.02):
    """
    Calculate maximum value in a pulse using the binned data

    Parameters
    ----------
    t_max
        Time above which the max search should start

    """
    print("Calculating maximum values from binned data")

    # Calculate max values for those quantities where binned data is to be used
    k = list(binned.keys())[0]
    for p in binned[k].pulse:
        for k, v in info.items():
            if v["max"] is None:
                continue
            max_search = xr.where(
                binned[k].t > t_max, binned[k].value.sel(pulse=p), np.nan
            )
            if not any(np.isfinite(max_search)):
                max_val[k].value.loc[dict(pulse=p)] = np.nan
                max_val[k].error.loc[dict(pulse=p)] = np.nan
                max_val[k].time.loc[dict(pulse=p)] = np.nan
                continue
            tind = max_search.argmax(dim="t", skipna=True).values
            tmax = binned[k].t[tind]
            max_val[k].time.loc[dict(pulse=p)] = tmax
            max_val[k].value.loc[dict(pulse=p)] = binned[k].value.sel(pulse=p, t=tmax)
            max_val[k].error.loc[dict(pulse=p)] = binned[k].error.sel(pulse=p, t=tmax)

    return max_val


def selection_criteria(binned, cond):
    """
    Find values within specified limits

    Parameters
    ----------
    binned
        Database binned result dictionary
    cond
        Dictionary of database keys with respective limits e.g.
        {"nirh1":{"var":"value", "lim":(0, 2.e19)}}
        where:
        - "nirh1" is the key of results dictionary
        - "var" is variable of the dataset to be used for the selection,
        either "value", "perc_error", "gradient", "norm_gradient"
        - "lim" = 2 element tuple with lower and upper limits

    Returns
    -------
        Boolean Dataarray of the same shape as the binned data with
        items == True if satisfying the selection criteria

    """

    k = list(cond.keys())[0]
    selection = xr.where(xr.ones_like(binned[k].value) == 1, True, False)
    for k, c in cond.items():
        item = binned[k]
        if c["var"] == "error":  # percentage error
            val = np.abs(item["error"] / item["value"])
        else:
            # val = flat(item[c["var"]])
            val = item[c["var"]]

        lim = c["lim"]
        if len(lim) == 1:
            selection *= val == lim
        else:
            if not np.isfinite(lim[0]):
                selection *= val < lim[1]
            elif not np.isfinite(lim[1]):
                selection *= val >= lim[0]
            else:
                selection *= (val >= lim[0]) * (val < lim[1])

    return selection


def flat(data: DataArray):
    return data.values.flatten()


def apply_selection(
    binned,
    cond=None,
    default=True,
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
            "Ohmic": {
                "nbi_power": {"var": "value", "lim": (0,)},
                "te0": {"var": "error", "lim": (np.nan, 0.2)},
                "ti0": {"var": "error", "lim": (np.nan, 0.2)},
            },
            "NBI": {
                "nbi_power": {"var": "value", "lim": (20, np.nan)},
                "te0": {"var": "error", "lim": (np.nan, 0.2)},
                "ti0": {"var": "error", "lim": (np.nan, 0.2)},
            },
        }

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
            filtered[kcond]["binned"] = binned_tmp
            filtered[kcond]["selection"] = selection_tmp
    else:
        filtered = {"All": {"selection": None, "binned": binned}}

    return filtered


def plot_time_evol(regr_data, info, to_plot, tplot=None, savefig=False, vlines=True):
    if tplot is None:
        tit_front = "Maximum of "
        tit_end = ""
        result = regr_data.max_val
    else:
        tit_front = ""
        tit_end = f" @ t={tplot:.3f} s"
        result = regr_data.binned

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
            yerr = flat(res.error)

            if len(ykey) > 1:
                label = info[k]["label"]
            else:
                label = ""

            plt.errorbar(
                x,
                yval,
                yerr=yerr,
                fmt="o",
                label=label,
                alpha=0.5,
            )
        plt.xlabel("Pulse #")
        plt.ylabel(info[k]["units"])
        plt.title(f"{tit_front}{title}{tit_end}")
        plt.xlim(np.min(regr_data.pulses)-5, np.max(regr_data.pulses)+5)
        plt.ylim(0,)
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
            save_figure(fig_name=name)


def plot_bivariate(
    filtered,
    info,
    to_plot,
    label=None,
    savefig=False,
):

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
                xval,
                yval,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                label=label,
                alpha=0.5,
            )
            plt.xlabel(xinfo["label"] + " " + xinfo["units"])
            plt.ylabel(yinfo["label"] + " " + yinfo["units"])
            plt.title(title)
        if label is not None:
            plt.legend()

        if savefig:
            save_figure(fig_name=name)


def plot_trivariate(
    filtered,
    info,
    to_plot,
    nbins=10,
    savefig=False,
):

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
                save_figure(fig_name=name)


def plot_hist(filtered, info, to_plot, tplot=None, bins=None, savefig=False):
    for title, ykey in to_plot.items():
        plt.figure()
        for i, key in enumerate(ykey):
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
            plt.hist(res, bins=bins, label=labels)
            plt.title(f"{info[key]['label']}")
            plt.xlabel(info[key]["units"])
            plt.legend()
            if tplot is not None:
                name += f"_t_{tplot:.3f}s"
            if savefig:
                save_figure(fig_name=name)


def plot(regr_data, filtered=None, tplot=0.03, default=True, savefig=False):

    info = regr_data.info

    ###################################
    # Simulated XRCS measurements
    ###################################
    plt.figure()
    temp_ratio = regr_data.temp_ratio
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
        save_figure(fig_name="XRCS_Te0_parametrization")

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
        save_figure(fig_name="XRCS_parametrization_temperatures")

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
        save_figure(fig_name="XRCS_parametrization_densities")

    ###################################
    # Time evolution of maximum quantities
    ###################################
    # Calculate RIP/IMC and add to values to be plotted

    # rip = regr_data.binned["rip_pfit"].sel(t=0.01, method="nearest").value
    # regr_data.max_val["rip_imc"] = deepcopy(regr_data.max_val["rip_pfit"])
    # regr_data.max_val["rip_imc"].value.values = (rip / (-regr_data.max_val["imc"] * 0.75 * 11).value).values
    to_plot = {
        "Electron Temperature": ("te_xrcs", "te0"),
        "Ion Temperature": ("ti_xrcs", "ti0"),
        "Electron Density": ("ne_nirh1",),
        "Stored Energy": ("wp_efit",),
        "Plasma Current": ("ipla_efit",),
        "MC Current": ("imc",),
        "Gas pressure": ("gas_press",),
    }
        # "Plasma current / MC current ": ("rip_imc",),
    plot_time_evol(regr_data, info, to_plot, savefig=savefig)

    ###################################
    # Time evolution of quantities at specified tplot
    ###################################
    to_plot = {
        "Bremsstrahlung PI": ("brems_pi",),
        "Bremsstrahlung MP": ("brems_mp",),
        "Plasma Current": ("ipla_efit",),
        "Gas pressure": ("gas_press",),
        "H-alpha": ("h_i_6563",),
        "Helium": ("he_ii_4686",),
        "Boron": ("b_ii_3451",),
        "Oxygen": ("o_iv_3063",),
        "Argon": ("ar_ii_4348",),
    }
    plot_time_evol(regr_data, info, to_plot, tplot=tplot, savefig=savefig)

    ###################################
    # Bivariate distributions for data-points which satisfy selection criteria
    ###################################
    if hasattr(regr_data, "filtered"):
        to_plot = {
            "T$_e$(0) vs. I$_P$": ("ipla_efit", "te0"),
            "T$_i$(0) vs. I$_P$": ("ipla_efit", "ti0"),
            "T$_i$(0) vs. n$_e$(NIRH1)": ("ne_nirh1", "ti0"),
            "T$_i$(0) vs. gas pressure": ("gas_press", "ti0"),
        }
        plot_bivariate(regr_data.filtered, info, to_plot, savefig=savefig)

    to_plot = {
        "Plasma Current": ("te0", "ti0", "ipla_efit"),
        "Electron Density": ("te0", "ti0", "ne_nirh1"),
        "Gas pressure": ("te0", "ti0", "gas_press"),
    }

    plot_hist(regr_data.filtered, info, to_plot, tplot=None, bins=None, savefig=savefig)

    filtered = deepcopy(regr_data.filtered)
    filtered["All"] = {"selection": None, "binned": regr_data.binned}
    plot_trivariate(filtered, info, to_plot, savefig=savefig)

    # (IP * RP) / (IMC * 0.75 * 11) at 10 ms vs. pulse #
    # plt.figure()
    # key = "imc_rip"
    # t_rip = binned[key].t.values
    # tr = np.int_((t_rip + [-regr_data.dt / 2, regr_data.dt / 2]) * 1.0e3)
    # xlab, ylab, tit = (
    #     "Pulse #",
    #     "RIP/IMC",
    #     f"(R*I$_P$ @ t={tr[0]}-{tr[1]} ms)" + " / (max(I$_{MC}$) * R$_{MC}$)",
    # )
    # val, err = flat(binned[key].value), flat(binned[key].error)
    # plt.errorbar(regr_data.pulses, val, yerr=err, fmt="o", label="")
    # plt.xlim(xlim[0], xlim[1])
    # add_to_plot(
    #     xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}_pulse", vlines=True
    # )

    # (IP * RP) / (IMC * 0.75 * 11) at 10 ms vs ITF
    # plt.figure()
    # key = "imc_rip"
    # itf = binned["itf"].sel(t=t_rip).value / 1.0e3
    # xlab, ylab, tit = (
    #     "I$_{TF}$ (kA)",
    #     "RIP/IMC",
    #     f"(R*I$_P$ @ t={tr[0]}-{tr[1]} ms)" + " / (max(I$_{MC}$) * R$_{MC}$)",
    # )
    # val, err = flat(binned[key].value), flat(binned[key].error)
    # plt.errorbar(itf, val, yerr=err, fmt="o", label="")
    # add_to_plot(
    #     xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}_itf",
    # )


def write_to_csv(regr_data):

    results = regr_data.filtered
    ipla_max = results["ipla_efit_max"].value.values
    ipla_max_time = results["ipla_efit_max"].time.values
    ti_max = results["ti_xrcs_max"].value.values
    ti_max_err = results["ti_xrcs_max"].error.values
    ti_max_time = results["ti_xrcs_max"].time.values
    ratio = regr_data.temp_ratio.sel(
        te_xrcs=results["te_xrcs_max"].value.values, method="nearest"
    ).values
    ti_0 = ti_max * ratio
    ipla_at_max = []
    nbi_at_max = []

    pulses = regr_data.pulses
    for i, p in enumerate(pulses):
        if np.isfinite(ti_max_time[i]):
            ipla = results["ipla_efit"].sel(pulse=p, t=ti_max_time[i]).value.values
            nbi = results["nbi_power"].sel(pulse=p, t=ti_max_time[i]).value.values
        else:
            ipla = np.nan
            nbi = np.nan
        ipla_at_max.append(ipla)
        nbi_at_max.append(nbi)

    ipla_at_max = np.array(ipla_at_max)
    nbi_at_max = np.array(nbi_at_max)

    to_write = {
        "pulse": pulses,
        "Ti max (eV)": ti_max,
        "Error of Ti max (eV)": ti_max_err,
        "Time (s) of Ti max": ti_max_time,
        "Ip (A) at time of max Ti": ipla_at_max,
        "NBI power (W) at time of max Ti": nbi_at_max,
        "Ip max (A)": ipla_max,
        "Time (s) of Ip max": ipla_max_time,
        "Ti (0) (keV)": ti_0,
    }
    df = pd.DataFrame(to_write)
    # df.to_csv()
    return df


def add_vlines(xvalues, color="k"):
    ylim = plt.ylim()
    for b in xvalues:
        plt.vlines(b, ylim[0], ylim[1], linestyles="dashed", colors=color, alpha=0.5)


def save_figure(fig_name="", orientation="landscape", ext=".jpg"):
    if getpass.getuser() == "lavoro":
        path = "/Users/lavoro/Work/Python/figures/regr_trends/"
    else:
        path = f"/home/{getpass.getuser()}/python/figures/regr_trends/"
    plt.savefig(
        path + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )


def simulate_xrcs():
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
    te_0 = np.linspace(0.5e3, 5.0e3, 50)  # central temperature
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


def write_to_pickle(regr_data):
    print(f"Saving regression database to \n {regr_data.path + regr_data.filename}")
    pickle.dump(regr_data, open(regr_data.path + regr_data.filename, "wb"))


def get_data_info():

    info = {
        "ipla_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".constraints.ip:cvalue",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "I$_P$ EFIT",
            "units": "(MA)",
            "const": 1.0e-6,
        },
        "wp_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".virial:wp",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "W$_P$ EFIT",
            "units": "(kJ)",
            "const": 1.0e-3,
        },
        "q95_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:q95",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "q$_{95}$ EFIT",
            "units": "",
            "const": 1.0,
        },
        "ipla_pfit": {
            "uid": "",
            "diag": "pfit",
            "node": ".post_best.results.global:ip",
            "seq": -1,
            "err": None,
            "max": True,
            "label": "I$_P$ PFIT",
            "units": "(MA)",
            "const": 1.0e-6,
        },
        "rip_pfit": {
            "uid": "",
            "diag": "pfit",
            "node": ".post_best.results.global:rip",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "W$_P$ EFIT",
            "units": "(kJ)",
            "const": 1.0e-3,
        },
        "vloop": {
            "uid": "",
            "diag": "mag",
            "node": ".floop.l026:v",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "V$_{loop}$ L026",
            "units": "(V)",
            "const": 1.0,
        },
        "ne_nirh1": {
            "uid": "interferom",
            "diag": "nirh1",
            "node": ".line_int:ne",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "n$_{e}$-int NIRH1",
            "units": "($10^{19}$ $m^{-3}$)",
            "const": 1.0e-19,
        },
        "ne_smmh1": {
            "uid": "interferom",
            "diag": "smmh1",
            "node": ".line_int:ne",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "n$_{e}$-int SMMH1",
            "units": "($10^{19}$ $m^{-3}$)",
            "const": 1.0e-19,
        },
        "gas_puff": {
            "uid": "",
            "diag": "gas",
            "node": ".puff_valve:gas_total",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "Total gas",
            "units": "(V)",
            "const": 1.0,
        },
        "gas_press": {
            "uid": "",
            "diag": "mcs",
            "node": ".mcs004:ch019",
            "seq": -1,
            "err": None,
            "max": True,
            "label": "Total Pressure",
            "units": "(a.u.)",
            "const": 1.0e3,
        },
        "imc": {
            "uid": "",
            "diag": "psu",
            "node": ".mc:i",
            "seq": -1,
            "err": None,
            "max": True,
            "label": "I$_{MC}$",
            "units": "(kA)",
            "const": 1.0e-3,
        },
        "itf": {
            "uid": "",
            "diag": "psu",
            "node": ".tf:i",
            "seq": -1,
            "err": None,
            "max": True,
            "label": "I$_{TF}$",
            "units": "(kA)",
            "const": 1.0e-3,
        },
        "brems_pi": {
            "uid": "spectrom",
            "diag": "princeton.passive",
            "node": ".dc:brem_mp",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "Bremsstrahlung PI",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "brems_mp": {
            "uid": "spectrom",
            "diag": "lines",
            "node": ".brem_mp1:intensity",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "Bremsstrahlung MP",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "te_xrcs": {
            "uid": "sxr",
            "diag": "xrcs",
            "node": ".te_kw:te",
            "seq": 0,
            "err": ".te_kw:te_err",
            "max": False,
            "label": "T$_e$ XRCS",
            "units": "(keV)",
            "const": 1.0e-3,
        },
        "ti_xrcs": {
            "uid": "sxr",
            "diag": "xrcs",
            "node": ".ti_w:ti",
            "seq": 0,
            "err": ".ti_w:ti_err",
            "max": False,
            "label": "T$_i$ XRCS",
            "units": "(keV)",
            "const": 1.0e-3,
        },
        "hnbi": {
            "uid": "raw_nbi",
            "diag": "hnbi1",
            "node": ".hv_ps:i_jema",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "P$_{HNBI}$",
            "units": "(a.u.)",
            "const": 1.0e-6,
        },
        "h_i_6563": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.h_i_6563:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "H I 656.3 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "he_ii_4686": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.he_ii_4686:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "He II 468.6 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "b_ii_3451": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.b_ii_3451:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "B II 345.1 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "o_iv_3063": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.o_iv_3063:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "O IV 306.3 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "ar_ii_4348": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.ar_ii_4348:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "Ar II 434.8 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "te0": {"max": False, "label": "T$_e$(0)", "units": "(keV)", "const": 1.0e-3},
        "ti0": {"max": False, "label": "T$_i$(0)", "units": "(keV)", "const": 1.0e-3},
        "nbi_power": {
            "max": False,
            "label": "P$_{NBI}$",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "gas_prefill": {
            "max": True,
            "label": "Gas prefill",
            "units": "(V * s)",
            "const": 1.0,
        },
        "gas_total": {
            "max": True,
            "label": "Total gas",
            "units": "(V * s)",
            "const": 1.0,
        },
        "rip_imc": {
            "max": None,
            "label": "(R$_{geo}$ I$_P$ @ 10 ms) / (R$_{MC}$ max(I$_{MC})$)",
            "units": " ",
            "const": 1.0,
        },
    }

    return info


def add_to_plot(xlab, ylab, tit, legend=True, vlines=False):
    if vlines:
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    if legend:
        plt.legend()
