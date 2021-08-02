"""Read and plot time evolution of various quantities
at identical times in a discharge over a defined pulse range

Example call:

    import hda.analyse_trends as trends
    corr = trends.correlations(8400, 8534, t=[0.03, 0.08])


TODO: add the following quantities
Ti/Te, Vloop, all NBI, li, betaP, geometry (volm, elongation, ..)
"""

import getpass
import numpy as np
from indica.readers import ST40Reader
import matplotlib.pylab as plt
import matplotlib.cm as cm
from copy import deepcopy
from xarray import DataArray, Dataset
import xarray as xr
import pickle

import pandas as pd
import hda.fac_profiles as fac
from hda.forward_models import Spectrometer
from indica.readers import ADASReader

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

        self.info = get_data_info()

        binned = {}
        max_val = {}
        for k in self.info.keys():
            binned[k] = []
            max_val[k] = []

        pulses_all = []
        for pulse in np.arange(pulse_start, pulse_end + 1):
            print(pulse)
            reader = ST40Reader(int(pulse), self.tlim[0], self.tlim[1],)

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
    to_zero = ["nbi_power"]
    for k in to_zero:
        results[k] = xr.where(
            (results[k] > 0) * np.isfinite(results[k]), results[k], 0,
        )

    to_nan = ["te_xrcs", "ti_xrcs", "ti0", "te0", "ipla_efit", "ipla_pfit", "wp_efit"]
    for k in to_nan:
        results[k] = xr.where(
            (results[k] > 0) * (np.isfinite(results[k])), results[k], np.nan,
        )

    return results


def calc_central_temperature(binned, temp_ratio):

    # Central temperatures from XRCS parametrization
    mult_binned = []
    profs = np.arange(len(temp_ratio))
    for i in range(len(temp_ratio)):
        mult_binned.append(
            temp_ratio[i].interp(te_xrcs=binned["te_xrcs"].value, method="linear")
        )
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
    # Calculate max values for those quantities where binned data is to be used
    for k, v in info.items():
        if v["max"] == True:
            continue
        for p in binned[k].pulse:
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
        Flattened array of Indices of elements satisfying the conditions

    """

    k = list(cond.keys())[0]
    selection = np.where(np.ones_like(flat(binned[k].value)) == 1, True, False)
    for k, c in cond.items():
        item = binned[k]
        if c["var"] == "error":  # percentage error
            val = flat(item["error"] / item["value"])
        else:
            val = flat(item[c["var"]])
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


def plot_time_evol(regr_data, tplot=None, savefig=False):

    to_plot = {
        "Electron Temperature": ("te_xrcs", "te0"),
        "Ion Temperature": ("ti_xrcs", "ti0"),
        "Bremsstrahlung PI": ("brems_pi",),
        "Bremsstrahlung MP": ("brems_mp",),
        "Stored Energy": ("wp_efit",),
        "Plasma Current": ("ipla_efit",),
        "MC Current": ("ipla_efit",),
        "TF Current": ("itf",),
        "Gas pressure": ("gas_press",),
        "H-alpha": ("h_i_6563",),
        "Helium": ("he_ii_4686",),
        "Boron": ("b_ii_3451",),
        "Oxygen": ("o_iv_3063",),
        "Argon": ("ar_ii_4348",),
    }

    info = regr_data.info
    if tplot is not None:
        result = regr_data.binned
        to_add = ""
    else:
        result = regr_data.max_val
        to_add = "Maximum "

    for title, ykey in to_plot.items():
        # TODO: add selection criteria to be applied to binned values and calculate max_val
        y = []
        for k in ykey:
            res = result[k] * info[k]["const"]
            if tplot is not None:
                res = res.sel(t=tplot)
            y.append(res)

        plt.figure()
        for i, k in enumerate(ykey):
            x = y[i].pulse
            yval = flat(y[i].value)
            yerr = flat(y[i].error)

            label = None
            if len(ykey) > 1:
                label = info[k]["label"]

            # ind = np.where(isel[i] == True)[0]
            ind = np.where(np.ones_like(yval) == 1)[0]
            plt.errorbar(
                x[ind], yval[ind], yerr=yerr[ind], fmt="o", label=label, alpha=0.5,
            )
        plt.xlabel("Pulse #")
        plt.ylabel(info[k]["units"])
        plt.title(f"{to_add}{title}")
        plt.ylim(0, )
        if label is not None:
            plt.legend()


def plot_trivariate(
    regr_data,
    xbins=None,
    ybins=None,
    zbins=None,
    savefig=False,
):

    to_plot = {
        "Plasma Current": ("te0", "ti0", "ipla_efit"),
        "Electron Density": ("te0", "ti0", "ne_nirh1"),
    }

    info = regr_data.info
    binned = regr_data.binned

    for title, keys in to_plot.items():
        xkey, ykey, zkey = keys
        xinfo = info[xkey]
        yinfo = info[ykey]
        zinfo = info[zkey]

        x = binned[xkey].value.values.flatten() * xinfo["const"]
        y = binned[ykey].value.values.flatten() * yinfo["const"]
        z = binned[zkey].value.values.flatten() * zinfo["const"]
        xerr = binned[xkey].error.values.flatten() * xinfo["const"]
        yerr = binned[ykey].error.values.flatten() * yinfo["const"]
        zerr = binned[zkey].error.values.flatten() * zinfo["const"]

        plt.figure()
        xhist = plt.hist(x, bins=xbins)
        plt.title(xinfo["label"])
        plt.xlabel(xinfo["units"])

        plt.figure()
        yhist = plt.hist(y, bins=ybins)
        plt.title(yinfo["label"])
        plt.xlabel(yinfo["units"])

        plt.figure()
        zhist = plt.hist(z, bins=zbins)
        plt.title(zinfo["label"])
        plt.xlabel(zinfo["units"])

        bins = zhist[1]
        bins_str = [f"{b:.1f}" for b in bins]
        nbins = len(bins) - 1
        cols = cm.rainbow(np.linspace(0, 1, nbins))
        plt.figure()
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
                alpha=0.5,
            )
        plt.xlabel(xinfo["label"] + " " + xinfo["units"])
        plt.ylabel(yinfo["label"] + " " + yinfo["units"])
        plt.title(f"{zinfo['label']} {zinfo['units']}")
        plt.legend()


def plot_bivariate_binned(
    regr_data,
    savefig=False,
    cond=None,
    plot_max=False,
):

    to_plot = {
        "Plasma Current": ("ipla_efit", "te0"),
    }

    binned = regr_data.binned
    info = regr_data.info

    for title, keys in to_plot.items():
        xkey, ykey, zkey = keys
        xinfo = info[xkey]
        yinfo = info[ykey]

        x = binned[xkey] * xinfo["const"]
        y = binned[ykey] * yinfo["const"]
        xval = flat(x.value)
        xerr = flat(x.error)
        yval = flat(y.value)
        yerr = flat(y.error)

        # isel = []
        # if cond is not None:
        #     for c in cond:
        #         isel.append(selection_criteria(binned, c))
        # else:
        #     isel.append(np.where(np.ones_like(xval) == 1, True, False))
        #
        # if label is None:
        #     label = [""] * len(isel)

        # Plot (filtered) binned data
        plt.figure()
        for i in range(len(isel)):
            # ind = np.where(isel[i] == True)[0]
            ind = np.where(np.ones_like(yval) == 1)[0]
            plt.errorbar(
                xval[ind],
                yval[ind],
                xerr=xerr[ind],
                yerr=yerr[ind],
                fmt="o",
                label=label[i],
                alpha=0.5,
            )
        plt.xlabel(xinfo["label"] + " " + xinfo["units"])
        plt.ylabel(yinfo["label"] + " " + yinfo["units"])
        plt.title(title)
        if label is not None:
            plt.legend()


def plot_evol(
    result,
    info,
    tplot=None,
    ylim_lo=None,
    ykey=("te0"),
    title="",
    label=None,
    savefig=False,
):

    if type(ykey) is str:
        ykey = (ykey,)



def plot(regr_data, tplot=0.03, savefig=False, xlim=()):

    binned = regr_data.binned
    max_val = regr_data.max_val

    # Simulate measurements
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
        savefig=savefig,
        name="XRCS_Te0_parametrization",
    )

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
        savefig=savefig,
        name="XRCS_parametrization_temperatures",
    )

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
        "T$_e$ (keV)",
        "Temperature profiles",
        savefig=savefig,
        name="XRCS_parametrization_densities",
    )

    if len(xlim) == 0:
        xlim = (
            regr_data.pulses.min() - 2,
            regr_data.pulses.max() + 2,
        )
    pulse_range = f"{xlim[0]}-{xlim[1]}"

    trange = f"t=[{regr_data.tlim[0]:1.3f}, {regr_data.tlim[1]:1.3f}] s"
    prange = f"pulses={pulse_range}"
    tplot = regr_data.time[np.argmin(np.abs(regr_data.time - tplot))]
    tplot_tit = f"t={tplot}"
    name = f"ST40_trends_{pulse_range}"

    plt.figure()
    key = "ipla_efit"
    if key in binned.keys():
        const = 1.0e-6
        xlab, ylab, lab, tit = ("Pulse #", "$(MA)$", "", f"Max I$_P$(EFIT)")
        val, err = (flat(max_val[key].value), flat(max_val[key].error))
        plt.errorbar(
            regr_data.pulses, val * const, yerr=err * const, fmt="o", label=lab
        )
        plt.ylim(0,)
        plt.xlim(xlim[0], xlim[1])
        add_to_plot(
            xlab,
            ylab,
            tit,
            legend=True,
            savefig=savefig,
            name=f"{name}_{key}",
            vlines=True,
        )

    plt.figure()
    key = "ipla_pfit"
    if key in binned.keys():
        const = 1.0e-6
        xlab, ylab, tit = ("Pulse #", "$(MA)$", f"Max I$_P$(PFIT)")
        val, err = (flat(max_val[key].value), flat(max_val[key].error))
        plt.errorbar(regr_data.pulses, val * const, yerr=err * const, fmt="*", label="")
        plt.ylim(0,)
        plt.xlim(xlim[0], xlim[1])
        add_to_plot(xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}", vlines=True)

    plt.figure()
    key = "te_xrcs"
    if key in binned.keys():
        xlab, ylab, tit = (
            "Pulse #",
            "(keV)",
            "Electron Temperature",
        )
        val, err = (flat(max_val[key].value), flat(max_val[key].error))
        const = 1.0e-3
        plt.errorbar(
            regr_data.pulses,
            val * const,
            yerr=err * const,
            fmt="o",
            label="T$_e$(XRCS)",
        )
        if "te0" in binned.keys():
            val, err = flat(max_val["te0"].value), flat(max_val["te0"].error)
            plt.errorbar(
                regr_data.pulses,
                val * const,
                yerr=err * const,
                fmt="o",
                label="T$_e$(0)",
            )
        plt.ylim(0,)
        plt.xlim(xlim[0], xlim[1])
        add_to_plot(xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}", vlines=True)

    plt.figure()
    key = "ti_xrcs"
    if key in binned.keys():
        xlab, ylab, tit = ("Pulse #", "$(keV)$", "Ion Temperature")
        val, err = flat(max_val[key].value), flat(max_val[key].error)
        const = 1.0e-3
        plt.errorbar(
            regr_data.pulses,
            val * const,
            yerr=err * const,
            fmt="o",
            label="T$_i$(XRCS)",
        )
        if "te0" in binned.keys():
            val, err = flat(max_val["ti0"].value), flat(max_val["ti0"].error)
            plt.errorbar(
                regr_data.pulses,
                val * const,
                yerr=err * const,
                fmt="o",
                label="T$_i$(0)",
            )
        plt.ylim(0,)
        plt.xlim(xlim[0], xlim[1])
        add_to_plot(xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}", vlines=True)

    plt.figure()
    key = "wp_efit"
    const = 1.0e-3
    xlab, ylab, tit = ("Pulse #", "$(kJ)$", "Stored Energy")
    val, err = flat(max_val[key].value), flat(max_val[key].error)
    plt.errorbar(
        regr_data.pulses, val * const, yerr=err * const, fmt="o", label="W$_P$(EFIT)"
    )
    plt.ylim(0,)
    plt.xlim(xlim[0], xlim[1])
    add_to_plot(xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}", vlines=True)

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

    # Indices for Ohmic and NBI phases

    binned = regr_data.binned
    info = regr_data.info
    cond = [
        {"nbi_power": {"var": "value", "lim": (0,)}},
        {"nbi_power": {"var": "value", "lim": (20, np.nan)}},
    ]
    label = ["Ohmic phases", "NBI phases"]
    plot_bivariate_binned(
        regr_data,
        xkey="ipla_efit",
        ykey="te0",
        savefig=False,
        title="",
        cond=cond,
        label=label,
    )

    return

    cond = {"nbi_power": {"var": "value", "lim": (0,)}}
    sel_bin_ohm, sel_max_ohm = self.selection_criteria(cond)
    ipla_val_ohm = (
        flat(xr.where(sel_bin_ohm, binned["ipla_efit"], np.nan).value) / 1.0e6
    )
    ipla_err_ohm = (
        flat(xr.where(sel_bin_ohm, binned["ipla_efit"], np.nan).error) / 1.0e6
    )
    te0_val_ohm = flat(xr.where(sel_bin_ohm, binned["te0"], np.nan).value) / 1.0e3
    te0_err_ohm = flat(xr.where(sel_bin_ohm, binned["te0"], np.nan).error) / 1.0e3

    cond = {"nbi_power": {"var": "value", "lim": (1, 1.0e6)}}
    sel_bin_nbi, sel_max_nbi = selection_criteria(cond)
    ipla_val_nbi = (
        flat(xr.where(sel_bin_nbi, binned["ipla_efit"], np.nan).value) / 1.0e6
    )
    ipla_err_nbi = (
        flat(xr.where(sel_bin_nbi, binned["ipla_efit"], np.nan).error) / 1.0e6
    )
    te0_val_nbi = flat(xr.where(sel_bin_nbi, binned["te0"], np.nan).value) / 1.0e3
    te0_err_nbi = flat(xr.where(sel_bin_nbi, binned["te0"], np.nan).error) / 1.0e3

    plt.figure()
    key = "te0_ipla"
    const = 1.0e-3
    xlab, ylab, tit = ("I$_{P}$ (MA)", "(keV)", "Central Electron Temperature")
    plt.errorbar(
        ipla_val_ohm,
        te0_val_ohm,
        yerr=te0_err_ohm,
        xerr=ipla_err_ohm,
        fmt="x",
        label=f"Ohmic phases",
    )
    plt.errorbar(
        ipla_val_nbi,
        te0_val_nbi,
        yerr=te0_err_nbi,
        xerr=ipla_err_nbi,
        fmt="o",
        label=f"NBI phases",
    )
    add_to_plot(
        xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}",
    )

    ti0_val_ohm = flat(xr.where(sel_bin_ohm, binned["ti0"], np.nan).value) / 1.0e3
    ti0_err_ohm = flat(xr.where(sel_bin_ohm, binned["ti0"], np.nan).error) / 1.0e3
    ti0_val_nbi = flat(xr.where(sel_bin_nbi, binned["ti0"], np.nan).value) / 1.0e3
    ti0_err_nbi = flat(xr.where(sel_bin_nbi, binned["ti0"], np.nan).error) / 1.0e3

    plt.figure()
    key = "ti0_ipla"
    const = 1.0e-3
    xlab, ylab, tit = ("I$_{P}$ (MA)", "(keV)", "Central Ion Temperature")
    plt.errorbar(
        ipla_val_ohm,
        ti0_val_ohm,
        yerr=ti0_err_ohm,
        xerr=ipla_err_ohm,
        fmt="x",
        label=f"Ohmic phases",
    )
    plt.errorbar(
        ipla_val_nbi,
        ti0_val_nbi,
        yerr=ti0_err_nbi,
        xerr=ipla_err_nbi,
        fmt="o",
        label=f"NBI phases",
    )
    add_to_plot(
        xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}",
    )

    # Temperature vs density
    nirh1_val = flat(binned["ne_nirh1"].value) / 1.0e19
    nirh1_err = flat(binned["ne_nirh1"].error) / 1.0e19

    val = flat(binned["te0"].value)
    err = flat(binned["te0"].error)
    plt.figure()
    key = "te0_nirh1_rescaled"
    const = 1.0e-3
    xlab, ylab, tit = (
        "N$_{e}$-int (10$^{19}$)",
        "(keV)",
        "Central Electron Temperature vs Density NIR",
    )
    plt.errorbar(
        nirh1_val[ind_ohm],
        (val * const)[ind_ohm],
        yerr=(err * const)[ind_ohm],
        xerr=nirh1_err[ind_ohm],
        fmt="x",
        label=f"Ohmic phases",
    )
    plt.errorbar(
        nirh1_val[ind_nbi],
        (val * const)[ind_nbi],
        yerr=(err * const)[ind_nbi],
        xerr=nirh1_err[ind_nbi],
        fmt="o",
        label=f"NBI phases",
    )
    add_to_plot(
        xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}",
    )

    val = flat(binned["ti0"].value)
    err = flat(binned["ti0"].error)
    plt.figure()
    key = "ti0_nirh1_rescaled"
    const = 1.0e-3
    xlab, ylab, tit = (
        "N$_{e}$-int (10$^{19}$)",
        "(keV)",
        "Central Ion Temperature vs Density NIR",
    )
    plt.errorbar(
        nirh1_val[ind_ohm],
        (val * const)[ind_ohm],
        yerr=(err * const)[ind_ohm],
        xerr=nirh1_err[ind_ohm],
        fmt="x",
        label=f"Ohmic phases",
    )
    plt.errorbar(
        nirh1_val[ind_nbi],
        (val * const)[ind_nbi],
        yerr=(err * const)[ind_nbi],
        xerr=nirh1_err[ind_nbi],
        fmt="o",
        label=f"NBI phases",
    )
    add_to_plot(
        xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}",
    )


def write_to_csv(regr_data):

    results = regr_data.filtered_data
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
    plt.savefig(
        f"/home/{getpass.getuser()}/python/figures/regr_trends/" + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )


def simulate_xrcs():

    adasreader = ADASReader()
    xrcs = Spectrometer(
        adasreader, "ar", "16", transition="(1)1(1.0)-(1)0(0.0)", wavelength=4.0,
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


def add_to_plot(xlab, ylab, tit, legend=True, name="", savefig=False, vlines=False):
    if vlines:
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    if legend:
        plt.legend()
    if savefig:
        save_figure(fig_name=name)


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
            "label": "T$_e$ XRCS",
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
            "node": ".line_evol.b_v_494:intensity",
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
            "node": ".line_evol.o_iv_306:intensity",
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
            "node": ".line_evol.ar_ii_443:intensity",
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
    }

    return info
