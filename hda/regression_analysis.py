"""Read and plot time evolution of various quantities
at identical times in a discharge over a defined pulse range

Example call:

    import hda.analyse_trends as trends
    corr = trends.correlations(8400, 8534, t=[0.03, 0.08])

"""

import getpass
import numpy as np
from indica.readers import ST40Reader
import matplotlib.pylab as plt
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
        pulse_start,
        pulse_end,
        tlim=(-0.03, 0.3),
        dt=0.01,
        overlap=0.5,
        reload=True,
    ):
        self.tlim = tlim

        self.dt = dt
        self.overlap = overlap

        self.time = np.arange(self.tlim[0], self.tlim[1], dt * overlap)
        self.empty = (
            np.full(self.time.shape, np.nan),
            np.full(self.time.shape, np.nan),
            np.full(self.time.shape, np.nan),
        )
        if reload:
            filename = f"/home/{getpass.getuser()}/python/regression_database.pkl"
            regr_data = pickle.load(open(filename, "rb"))
            self.tlim = regr_data.tlim
            self.dt = regr_data.dt
            self.overlap = regr_data.overlap
            self.time = regr_data.time
            self.empty = regr_data.empty
            self.binned_data = regr_data.binned_data
            self.filtered_data = regr_data.filtered_data
            self.lines_labels = regr_data.lines_labels
            self.pulses = regr_data.pulses
        else:
            self.binned_data = self.read_data(pulse_start, pulse_end)

        el_temp, xrcs, temp_ratio = simulate_xrcs()
        xrcs.input_te = el_temp
        self.temp_ratio = temp_ratio
        self.xrcs_sim = xrcs

    def __call__(self, *args, **kwargs):
        """
        Analyse last 150 pulses

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """

    def init_avantes(self):

        lines_labels = [
            "h_i_656",
            "he_ii_469",
            "b_v_494",
            "o_iv_306",
            "ar_ii_443",
        ]

        self.lines_labels = lines_labels

    def init_results_dict(self):
        results = {
            "pulses": [],
            "gas_prefill": [],
            "gas_total": [],
            "ipla_efit": [],
            "ipla_pfit": [],
            "wp_efit": [],
            "rip_pfit": [],
            "imc": [],
            "imc_max": [],
            "itf": [],
            "nirh1": [],
            "smmh1": [],
            "brems_pi": [],
            "brems_mp": [],
            "te_xrcs": [],
            "ti_xrcs": [],
            "nbi_power": [],
            "lines": [],
        }
        values = deepcopy(results)
        errors = deepcopy(results)
        gradients = deepcopy(results)

        return values, errors, gradients

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
        values, errors, gradients = self.init_results_dict()

        self.init_avantes()

        for pulse in np.arange(pulse_start, pulse_end + 1):
            print(pulse)
            reader = ST40Reader(int(pulse), self.tlim[0], self.tlim[1],)
            pfit = self.get_pfit(reader)
            efit = self.get_efit(reader)

            if not np.any(np.isfinite(pfit["ipla"][0])):
                continue
            if len(np.where(pfit["ipla"][0] > 0.1e6)[0]) == 0:
                continue
            if len(np.where(efit["ipla"][0] > 0.1e6)[0]) == 0:
                continue

            values["pulses"].append(pulse)  # np.array([pulse]*len(self.time)))

            values["ipla_efit"].append(efit["ipla"][0])
            errors["ipla_efit"].append(efit["ipla"][1])
            gradients["ipla_efit"].append(efit["ipla"][2])

            values["wp_efit"].append(efit["wp"][0])
            errors["wp_efit"].append(efit["wp"][1])
            gradients["wp_efit"].append(efit["wp"][2])

            values["ipla_pfit"].append(pfit["ipla"][0])
            errors["ipla_pfit"].append(pfit["ipla"][1])
            gradients["ipla_pfit"].append(pfit["ipla"][2])

            values["rip_pfit"].append(pfit["rip"][0])
            errors["rip_pfit"].append(pfit["rip"][1])
            gradients["rip_pfit"].append(pfit["rip"][2])

            mc = self.get_mc(reader)
            values["imc"].append(mc["imc"][0])
            errors["imc"].append(mc["imc"][1])
            gradients["imc"].append(mc["imc"][2])

            values["imc_max"].append(mc["imc_max"])

            tf = self.get_tf(reader)
            values["itf"].append(tf["itf"][0])
            errors["itf"].append(tf["itf"][1])
            gradients["itf"].append(tf["itf"][2])

            nirh1 = self.get_nirh1(reader)
            values["nirh1"].append(nirh1["ne_los_int"][0])
            errors["nirh1"].append(nirh1["ne_los_int"][1])
            gradients["nirh1"].append(nirh1["ne_los_int"][2])

            smmh1 = self.get_smmh1(reader)
            values["smmh1"].append(smmh1["ne_los_int"][0])
            errors["smmh1"].append(smmh1["ne_los_int"][1])
            gradients["smmh1"].append(smmh1["ne_los_int"][2])

            brems = self.get_brems(reader)
            values["brems_pi"].append(brems["brems_pi"][0])
            errors["brems_pi"].append(brems["brems_pi"][1])
            values["brems_mp"].append(brems["brems_mp"][0])
            errors["brems_mp"].append(brems["brems_mp"][1])

            xrcs = self.get_xrcs(reader, debug=False)
            values["te_xrcs"].append(xrcs["te"][0])
            errors["te_xrcs"].append(xrcs["te"][1])
            gradients["te_xrcs"].append(xrcs["te"][2])

            values["ti_xrcs"].append(xrcs["ti"][0])
            errors["ti_xrcs"].append(xrcs["ti"][1])
            gradients["ti_xrcs"].append(xrcs["ti"][2])

            nbi = self.get_nbi(reader)
            values["nbi_power"].append(nbi["power"][0])
            errors["nbi_power"].append(nbi["power"][1])

            gas = self.get_gas(reader)
            values["gas_prefill"].append(gas["prefill"])
            values["gas_total"].append(gas["cumul"])

            avantes = self.get_avantes(reader)
            values["lines"].append(avantes[0])
            errors["lines"].append(avantes[1])

        # Reorder data and make DataArray/Dataset,
        # calculate gradient for future selection criteria
        # return values

        self.pulses = np.unique(np.array(values["pulses"]))
        results = {}
        for k in values.keys():
            coords = [("pulse", values["pulses"])]
            values[k] = np.array(values[k])
            errors[k] = np.array(errors[k])
            gradients[k] = np.array(gradients[k])
            if len(values[k].shape) > 1:
                coords = [("pulse", values["pulses"]), ("t", self.time)]
            if len(values[k].shape) > 2:
                coords = [
                    ("pulse", values["pulses"]),
                    ("line", self.lines_labels),
                    ("t", self.time),
                ]

            value = DataArray(values[k], coords=coords)
            error = xr.full_like(value, 0)
            if errors[k].shape == values[k].shape:
                error = DataArray(errors[k], coords=coords)
            gradient = xr.full_like(value, np.nan)
            if gradients[k].shape == values[k].shape:
                gradient = DataArray(gradients[k], coords=coords)
            results[k] = Dataset({"value": value, "error": error, "gradient": gradient})

        del results["pulses"]
        return results

    def bin_in_time(self, data, time, err=None, debug=False):
        return bin_in_time(data, time, self.time, self.overlap, err=err, debug=debug)

    def add_data(self, pulse_end):
        """
        Add data from newer pulses to binned_data dictionary

        Parameters
        ----------
        pulse_end
            Last pulse to include in the analysis
        """
        pulse_start = np.array(self.results["pulses"]).max() + 1
        if pulse_end < pulse_start:
            print("Newer pulses only (for the time being...)")
            return
        new = self.read_data(pulse_start, pulse_end)

        for i, pulse in enumerate(new["pulses"]):
            for k1, res in new.items():
                if k1 == "pulses":
                    continue
                if type(res) != dict:
                    self.results[k1].append(res[i])
                    continue

                for k2, res2 in res.items():
                    self.results[k1][k2].append(res2[i])

    def get_efit(self, reader):
        res = {
            "ipla": deepcopy(self.empty),
            "wp": deepcopy(self.empty),
        }

        time, _ = reader._get_signal("", "efit", ":time", 0)
        if np.array_equal(time, "FAILED") or len(time) < 3:
            print("no Ip from EFIT")
            return res

        if np.min(time) > np.max(self.tlim) or np.max(time) < np.min(self.tlim):
            print("no Ip from EFIT in time range")
            return res

        ipla, _ = reader._get_signal("", "efit", ".constraints.ip:cvalue", 0)
        wp, _ = reader._get_signal("", "efit", ".virial:wp", 0)
        # volm, _ = reader._get_signal("", "efit", ".global:volm", 0)
        return {
            "ipla": self.bin_in_time(ipla, time),
            "wp": self.bin_in_time(wp, time),
        }

    def get_pfit(self, reader):
        res = {
            "ipla": deepcopy(self.empty),
            "rip": deepcopy(self.empty),
        }

        time, _ = reader._get_signal("", "pfit", ".post_best.results:time", -1)
        if np.array_equal(time, "FAILED"):
            print("no Ip from PFIT")
            return res

        if np.min(time) > np.max(self.tlim) or np.max(time) < np.min(self.tlim):
            print("no Ip from PFIT in time range")
            return res

        ipla, _ = reader._get_signal("", "pfit", ".post_best.results.global:ip", -1)
        rip, _ = reader._get_signal("", "pfit", ".post_best.results.global:rip", -1)
        if np.array_equal(ipla, "FAILED") or np.array_equal(ipla, "FAILED"):
            print("no Ip from PFIT")
            return res
        return {
            "ipla": self.bin_in_time(ipla, time),
            "rip": self.bin_in_time(rip, time),
        }

    def get_nirh1(self, reader):
        res = {"ne_los_int": deepcopy(self.empty)}
        ne_los_int, _path = reader._get_signal("interferom", "nirh1", ".line_int:ne", 0)
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED") and not np.array_equal(
            ne_los_int, "FAILED"
        ):
            time = time[0]
            res = {"ne_los_int": self.bin_in_time(ne_los_int, time)}
        return res

    def get_smmh1(self, reader):
        res = {"ne_los_int": deepcopy(self.empty)}

        ne_los_int, _path = reader._get_signal("interferom", "smmh1", ".line_int:ne", 0)
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED") and not np.array_equal(
            ne_los_int, "FAILED"
        ):
            time = time[0]
            res = {"ne_los_int": self.bin_in_time(ne_los_int, time)}
        return res

    def get_gas(self, reader):
        res = {"puff": deepcopy(self.empty)}
        puff, _path = reader._get_signal("", "gas", ".puff_valve:gas_total", -1)
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time[0], "FAILED") and not np.array_equal(puff, "FAILED"):
            time = time[0]
            dt = time[1] - time[0]
            it = np.where(time <= 0)
            res = {
                "prefill": np.sum(puff[it] * dt),
                "cumul": np.interp(self.time, time, np.cumsum(puff) * dt),
            }
        return res

    def get_mc(self, reader):
        res = {"imc": deepcopy(self.empty)}
        data, _path = reader._get_signal("", "psu", ".mc:i", -1)
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED") and not np.array_equal(data, "FAILED"):
            time = time[0]
            res = {"imc": self.bin_in_time(data, time), "imc_max": np.nanmax(data)}
        return res

    def get_tf(self, reader):
        res = {"itf": deepcopy(self.empty)}
        data, _path = reader._get_signal("", "psu", ".tf:i", -1)
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED") and not np.array_equal(data, "FAILED"):
            time = time[0]
            res = {"itf": self.bin_in_time(data, time)}
        return res

    def get_brems(self, reader):
        res = {"brems_pi": deepcopy(self.empty), "brems_mp": deepcopy(self.empty)}

        brems, _path = reader._get_signal(
            "spectrom", "princeton.passive", ".dc:brem_mp", 0
        )
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED") and not np.array_equal(brems, "FAILED"):
            time = time[0]
            res["brems_pi"] = self.bin_in_time(brems, time)

        brems, _path = reader._get_signal(
            "spectrom", "lines", ".brem_mp1:intensity", -1
        )
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED") and not np.array_equal(brems, "FAILED"):
            time = time[0]
            res["brems_mp"] = self.bin_in_time(brems, time)

        return res

    def get_xrcs(self, reader, debug=False):
        res = {"te": deepcopy(self.empty), "ti": deepcopy(self.empty)}

        ti, _path = reader._get_signal("sxr", "xrcs", ".ti_w:ti", 0)
        ti_err, _path = reader._get_signal("sxr", "xrcs", ".ti_w:ti_err", 0)
        te, path = reader._get_signal("sxr", "xrcs", ".te_kw:te", 0)
        te_err, path = reader._get_signal("sxr", "xrcs", ".te_kw:te_err", 0)
        time, _ = reader._get_signal_dims(_path, 1)

        if not np.array_equal(time, "FAILED") and not np.array_equal(te, "FAILED"):
            time = time[0]
            res = {
                "te": self.bin_in_time(te, time, err=te_err, debug=debug),
                "ti": self.bin_in_time(ti, time, err=ti_err, debug=debug),
            }
        return res

    def get_nbi(self, reader):
        i_hnbi, _path = reader._get_signal("raw_nbi", "hnbi1", ".hv_ps:i_jema", -1)
        time, _ = reader._get_signal_dims(_path, 1)
        v_hnbi = 1.0

        power = i_hnbi * v_hnbi
        time = time[0]
        return {"power": self.bin_in_time(power, time)}

    def get_avantes(self, reader):
        lines_avrg = []
        lines_stdev = []
        line_time, _ = reader._get_signal("spectrom", "avantes.line_mon", ":time", 0)
        for line in self.lines_labels:
            line_data, _ = reader._get_signal(
                "spectrom", "avantes.line_mon", f".line_evol.{line}:intensity", 0,
            )
            avrg, std, _ = self.bin_in_time(line_data, line_time)

            lines_avrg.append(avrg)
            lines_stdev.append(std)

        return np.array(lines_avrg), np.array(lines_stdev)


def filter_data(
    regr_data, err_perc=0.1, positive=True, t_rip=0.01, t_max=0.01, limits={}
):
    """
    Apply general selection criteria to filter data accordingly

    Parameters
    ----------
    err_perc
        Percentage error to exclude (stationarity & SNR)
    positive
        Exclude all negative values
    t_rip
        time at which RIP is to be chosen for comparison with I_MC
    t_max
        time from which maximum values should be searched
    """

    # Filter data before calculating extra values
    results = deepcopy(regr_data.binned_data)
    for k in results.keys():
        if positive:
            cond = results[k].value > 0
            results[k] = xr.where(cond, results[k], 0.0)
        if err_perc > 0:
            cond = (results[k].error / results[k].value) < err_perc
            results[k] = xr.where(cond, results[k], 0.0)

    tmp = DataArray(
        np.full(regr_data.pulses.shape, np.nan), coords=[("pulse", regr_data.pulses)],
    )
    tmp = Dataset(
        {"value": deepcopy(tmp), "error": deepcopy(tmp), "time": deepcopy(tmp),}
    )

    keys = ["ipla_efit", "ipla_pfit", "wp_efit", "te_xrcs", "ti_xrcs"]
    # TODO: find a way to assign error variable filtering on time for different pulses
    for k in keys:
        kmax = f"{k}_max"
        results[kmax] = deepcopy(tmp)
        max_search = xr.where(regr_data.time > t_max, results[k].value, np.nan)
        tind = max_search.argmax(dim="t", skipna=True).values
        tmax = regr_data.time[tind]
        results[kmax].value.values = results[k].value.max(dim="t", skipna=True)
        for ip, p in enumerate(regr_data.pulses):
            results[kmax].error.loc[dict(pulse=p)] = results[k].error.sel(
                pulse=p, t=tmax[ip]
            )
        results[kmax].time.values = tmax

    it = np.where(regr_data.time >= t_rip)[0][0]
    t_rip = regr_data.time[it]
    imc_rip = results["rip_pfit"].sel(t=t_rip) / (results["imc_max"] * 0.75 * 11)
    imc_rip.error.values = (
        results["rip_pfit"].error.sel(t=t_rip) / (results["imc_max"] * 0.75 * 11).value
    ).values
    results["imc_rip"] = imc_rip

    for k in results.keys():
        if positive:
            cond = results[k].value > 0
            results[k] = xr.where(cond, results[k], np.nan)
        if err_perc > 0:
            cond = (results[k].error / results[k].value) < err_perc
            results[k] = xr.where(cond, results[k], np.nan)

    results["nbi_power"] = xr.where(results["nbi_power"] > 0, results["nbi_power"], 0)

    regr_data.filtered_data = results

    return regr_data


def selection_criteria(results: Database, cond: dict, ind=False):
    """
    Find values within specified limits

    Parameters
    ----------
    results
        Binned_data dictionary of datasets containing the database data
    cond
        Dictionary of database keys with respective limits e.g.
        {"nirh1":{"var":"value", "lim":(0, 2.e19)}}
        where:
        - "nirh1" is the key of results dictionary
        - "var" is variable of the dataset to be used for the selection,
        either "value", "perc_error", "gradient", "norm_gradient"
        - "lim" = 2 element tuple with lower and upper limits
    ind
        True to flatten and return indices

    Returns
    -------
        Dictionary with the same keys as cond & containing the boolean array
        of the same structure as "var", satisfying the selection criteria

    """
    selection = None
    for k, c in cond.items():
        val = getattr(results[k], c["var"]).values.flatten()
        if len(c["lim"]) == 1:
            sel = val == c["lim"][0]
        else:
            sel = (val >= c["lim"][0]) * (val < c["lim"][1])
        if selection is None:
            selection = sel
        else:
            selection *= sel

    if ind:
        selection = np.where(selection)[0]

    return selection


def flat(data: DataArray):
    return data.values.flatten()


def plot(regr_data, tplot=0.03, savefig=False, xlim=()):

    if not hasattr(regr_data, "filtered_data"):
        print("Apply selection criteria before plotting...")
        return

    # Simulate measurements
    plt.figure()
    te0 = regr_data.xrcs_sim.input_te.sel(rho_poloidal=0).values
    plt.plot(te0, regr_data.xrcs_sim.el_temp.values, label="XRCS measurement")
    plt.plot(te0, te0, "--k", label="Central Te")
    plt.legend()
    add_to_plot(
        "T$_e$(0)",
        "T$_{e,i}$(XRCS)",
        "XRCS measurement vs. Central Te",
        savefig=savefig,
        name="XRCS_Te0",
    )

    results = regr_data.filtered_data
    mult_te_max = regr_data.temp_ratio.sel(
        te_xrcs=flat(results["te_xrcs_max"].value), method="nearest"
    ).values
    mult_te = regr_data.temp_ratio.sel(
        te_xrcs=flat(results["te_xrcs"].value), method="nearest"
    ).values

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
    key = "ipla_efit_max"
    const = 1.0e-6
    xlab, ylab, lab, tit = ("Pulse #", "$(MA)$", "", f"Max I$_P$(EFIT)")
    val, err = (flat(results[key].value), flat(results[key].error))
    plt.errorbar(regr_data.pulses, val * const, yerr=err * const, fmt="o", label=lab)
    plt.ylim(0,)
    plt.xlim(xlim[0], xlim[1])
    add_to_plot(
        xlab, ylab, tit, legend=True, savefig=savefig, name=f"{name}_{key}", vlines=True
    )

    plt.figure()
    key = "ipla_pfit_max"
    const = 1.0e-6
    xlab, ylab, tit = ("Pulse #", "$(MA)$", f"Max I$_P$(PFIT)")
    val, err = (flat(results[key].value), flat(results[key].error))
    plt.errorbar(regr_data.pulses, val * const, yerr=err * const, fmt="*", label="")
    plt.ylim(0,)
    plt.xlim(xlim[0], xlim[1])
    add_to_plot(xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}", vlines=True)

    plt.figure()
    key = "te_xrcs_max"
    xlab, ylab, tit = (
        "Pulse #",
        "(keV)",
        "Maximum Electron Temperature",
    )
    val, err = (flat(results[key].value), flat(results[key].error))
    const = 1.0e-3
    plt.errorbar(
        regr_data.pulses, val * const, yerr=err * const, fmt="o", label="T$_e$(XRCS)"
    )
    const = 1.0e-3 * mult_te_max
    plt.errorbar(
        regr_data.pulses, val * const, yerr=err * const, fmt="o", label="Max T$_e$(0)",
    )
    plt.ylim(0,)
    plt.xlim(xlim[0], xlim[1])
    add_to_plot(xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}", vlines=True)

    plt.figure()
    key = "ti_xrcs_max"
    xlab, ylab, tit = ("Pulse #", "$(keV)$", "Maximum Ion Temperature")
    val, err = flat(results[key].value), flat(results[key].error)
    const = 1.0e-3
    plt.errorbar(
        regr_data.pulses, val * const, yerr=err * const, fmt="o", label="T$_i$(XRCS)"
    )
    const = 1.0e-3 * mult_te_max
    plt.errorbar(
        regr_data.pulses, val * const, yerr=err * const, fmt="o", label="T$_i$(0)",
    )
    plt.ylim(0,)
    plt.xlim(xlim[0], xlim[1])
    add_to_plot(xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}", vlines=True)

    plt.figure()
    key = "wp_efit_max"
    const = 1.0e-3
    xlab, ylab, tit = ("Pulse #", "$(kJ)$", "Stored Energy")
    val, err = flat(results[key].value), flat(results[key].error)
    plt.errorbar(
        regr_data.pulses, val * const, yerr=err * const, fmt="o", label="W$_P$(EFIT)"
    )
    plt.ylim(0,)
    plt.xlim(xlim[0], xlim[1])
    add_to_plot(xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}", vlines=True)

    # (IP * RP) / (IMC * 0.75 * 11) at 10 ms vs. pulse #
    plt.figure()
    key = "imc_rip"
    t_rip = results[key].t.values
    tr = np.int_((t_rip + [-regr_data.dt / 2, regr_data.dt / 2]) * 1.0e3)
    xlab, ylab, tit = (
        "Pulse #",
        "RIP/IMC",
        f"(R*I$_P$ @ t={tr[0]}-{tr[1]} ms)" + " / (max(I$_{MC}$) * R$_{MC}$)",
    )
    val, err = flat(results[key].value), flat(results[key].error)
    plt.errorbar(regr_data.pulses, val, yerr=err, fmt="o", label="")
    plt.xlim(xlim[0], xlim[1])
    add_to_plot(
        xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}_pulse", vlines=True
    )

    # (IP * RP) / (IMC * 0.75 * 11) at 10 ms vs ITF
    plt.figure()
    key = "imc_rip"
    itf = results["itf"].sel(t=t_rip).value / 1.0e3
    xlab, ylab, tit = (
        "I$_{TF}$ (kA)",
        "RIP/IMC",
        f"(R*I$_P$ @ t={tr[0]}-{tr[1]} ms)" + " / (max(I$_{MC}$) * R$_{MC}$)",
    )
    val, err = flat(results[key].value), flat(results[key].error)
    plt.errorbar(itf, val, yerr=err, fmt="o", label="")
    add_to_plot(
        xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}_itf",
    )

    # Indices for Ohmic and NBI phases
    cond = {"nbi_power": {"var": "value", "lim": (0,)}}
    ind_ohm = selection_criteria(results, cond, ind=True)

    cond = {"nbi_power": {"var": "value", "lim": (1, 1.0e6)}}
    ind_nbi = selection_criteria(results, cond, ind=True)

    ipla_val = results["ipla_efit"].value.values.flatten() / 1.0e6
    ipla_err = results["ipla_efit"].error.values.flatten() / 1.0e6

    plt.figure()
    key = "te_ipla"
    const = 1.0e-3
    val = flat(results["te_xrcs"].value)
    err = flat(results["te_xrcs"].error)
    xlab, ylab, tit = ("I$_{P}$ (MA)", "(keV)", "Electron Temperature XRCS")
    plt.errorbar(
        ipla_val[ind_ohm],
        (val * const)[ind_ohm],
        yerr=(err * const)[ind_ohm],
        xerr=ipla_err[ind_ohm],
        fmt="x",
        label=f"Ohmic phases",
    )
    plt.errorbar(
        ipla_val[ind_nbi],
        (val * const)[ind_nbi],
        yerr=(err * const)[ind_nbi],
        xerr=ipla_err[ind_nbi],
        fmt="o",
        label=f"NBI phases",
    )
    add_to_plot(
        xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}",
    )

    plt.figure()
    key = "te_ipla_rescaled"
    const = 1.0e-3 * mult_te
    tit = "Central Electron Temperature"
    plt.errorbar(
        ipla_val[ind_ohm],
        (val * const)[ind_ohm],
        yerr=(err * const)[ind_ohm],
        xerr=ipla_err[ind_ohm],
        fmt="x",
        label=f"Ohmic phases",
    )
    plt.errorbar(
        ipla_val[ind_nbi],
        (val * const)[ind_nbi],
        yerr=(err * const)[ind_nbi],
        xerr=ipla_err[ind_nbi],
        fmt="o",
        label=f"NBI phases",
    )
    add_to_plot(
        xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}",
    )

    plt.figure()
    key = "ti_ipla"
    const = 1.0e-3
    val = flat(results["ti_xrcs"].value)
    err = flat(results["ti_xrcs"].error)
    xlab, ylab, tit = ("I$_{P}$ (MA)", "(keV)", "Ion Temperature XRCS")
    plt.errorbar(
        ipla_val[ind_ohm],
        (val * const)[ind_ohm],
        yerr=(err * const)[ind_ohm],
        xerr=ipla_err[ind_ohm],
        fmt="x",
        label=f"Ohmic phases",
    )
    plt.errorbar(
        ipla_val[ind_nbi],
        (val * const)[ind_nbi],
        yerr=(err * const)[ind_nbi],
        xerr=ipla_err[ind_nbi],
        fmt="o",
        label=f"NBI phases",
    )
    add_to_plot(
        xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}",
    )

    plt.figure()
    key = "ti_ipla_rescaled"
    const = 1.0e-3 * mult_te
    tit = "Central Ion Temperature"
    plt.errorbar(
        ipla_val[ind_ohm],
        (val * const)[ind_ohm],
        yerr=(err * const)[ind_ohm],
        xerr=ipla_err[ind_ohm],
        fmt="x",
        label=f"Ohmic phases",
    )
    plt.errorbar(
        ipla_val[ind_nbi],
        (val * const)[ind_nbi],
        yerr=(err * const)[ind_nbi],
        xerr=ipla_err[ind_nbi],
        fmt="o",
        label=f"NBI phases",
    )
    add_to_plot(
        xlab, ylab, tit, savefig=savefig, name=f"{name}_{key}",
    )

    # Ti(XRCS) vs density
    nirh1_val = flat(results["nirh1"].value) / 1.0e19
    nirh1_err = flat(results["nirh1"].error) / 1.0e19

    plt.figure()
    key = "ti_nirh1_rescaled"
    const = 1.0e-3 * mult_te
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

    # Ti vs Ip compared to modelling
    # select NBI pulses only with NIRH1 in range of highest Ti
    modelling = {
        "h_iter_old": {
            "ipla": np.array([500000, 1000000, 1500000]) / 1.0e6,
            "ti0_M": np.array([31.63, 54.65, 70.70]),
            "label": "H(ITER) old",
        },
        "h_pb_old": {
            "ipla": np.array([500000, 1000000, 1500000]) / 1.0e6,
            "ti0_M": np.array([77.67, 104.77, 133.37]),
            "label": "H(PB) old",
        },
        "h_iter_new": {
            "ipla": np.array([500000, 750000, 1000000]) / 1.0e6,
            "ti0_M": np.array([28.72, 40.39, 51.26]),
            "label": "H(ITER) new",
        },
        "h_pb_new": {
            "ipla": np.array([500000, 750000, 1000000]) / 1.0e6,
            "ti0_M": np.array([47.56, 67.23, 75.12]),
            "label": "H(PB) new",
        },
    }
    plt.figure()
    key = "ti_model"
    const = 1.0e-3 * 11.628 * mult_te
    val = flat(results["ti_xrcs"].value)
    err = flat(results["ti_xrcs"].error)
    xlab, ylab, tit = (
        "I$_{P}$ (MA)",
        "T$_i$ (M $^o$C)",
        f"Central ion temperature vs. Ip",
    )
    plt.errorbar(
        ipla_val,
        (val * const),
        yerr=(err * const),
        xerr=ipla_err,
        fmt="x",
        label=f"All data",
    )
    cond = {
        "nbi_power": {"var": "value", "lim": (35, 1.0e6)},
        "nirh1": {"var": "value", "lim": (0.5e20, 1.5e20)},
    }
    ind = selection_criteria(results, cond, ind=True)
    plt.errorbar(
        ipla_val[ind],
        (val * const)[ind],
        yerr=(err * const)[ind],
        xerr=ipla_err[ind],
        fmt="o",
        label=f"NBI & NIR =[0.5,1.5]e20",
    )
    for k in modelling.keys():
        plt.plot(
            modelling[k]["ipla"],
            modelling[k]["ti0_M"],
            label=modelling[k]["label"],
            marker="s",
        )

    plt.xlim(0,)
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

    rho = np.linspace(0, 1, 100)
    profs = fac.Plasma_profs(rho)
    profs.te /= profs.te[0]
    t = np.linspace(0, 1, 15)
    te0 = np.linspace(0.5e3, 5.0e3, 15)
    te_prof = []
    for te in te0:
        te_prof.append(profs.te * te)

    el_temp = xr.concat(te_prof, "t")
    el_temp = el_temp.assign_coords({"t": t})

    ti_const = [1.5, 1, 0.5]
    for const in ti_const:
        ion_temp = deepcopy(el_temp) * const

        xrcs.simulate_measurements(5.0e19, el_temp, ion_temp)

        temp_ratio = DataArray(
            te0 / xrcs.el_temp.values, coords=[("te_xrcs", xrcs.el_temp.values)]
        )
        #
        # plt.figure()
        # te0 = el_temp.sel(rho_poloidal=0).values
        # ti0 = ion_temp.sel(rho_poloidal=0).values
        # plt.plot(te0, xrcs.ion_temp.values, label="XRCS measurement")
        # plt.plot(te0, ti0, "--k", label="Central Ti")
        # plt.legend()
        # add_to_plot(
        #     xlab="T$_e$(0)",
        #     ylab="T$_i$(XRCS)",
        #     tit="XRCS measurement vs. Central Te",
        #     legend=True,
        #     savefig=False,
        #     name="XRCS_Te0",
        # )

    return el_temp, xrcs, temp_ratio


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


def bin_in_time(data, time, time_new, overlap=0.5, err=None, debug=False):

    dt = (time_new[1] - time_new[0]) / overlap
    ifin = np.where(np.isfinite(data))[0]
    if len(ifin) < 2:
        return (
            np.zeros_like(time_new) * np.nan,
            np.zeros_like(time_new) * np.nan,
            np.zeros_like(time_new) * np.nan,
        )

    if debug:
        print(time)
        print(data)
        plt.plot(time, data)
        # plt.plot(time, np.gradient(data, time))

    # gradient = np.gradient(data, time)
    binned_data = []
    binned_grad = []
    binned_err = []
    for t in time_new:
        _avrg = np.nan
        _grad = np.nan
        _stdev = 0.0

        tind = (time >= t - dt / 2.0) * (time < t + dt / 2.0)
        if len(tind) > 0:
            data_tmp = data[tind]
            ifin = np.where(np.isfinite(data_tmp))[0]
            if len(ifin) >= 1:
                _avrg = np.mean(data_tmp[ifin])
                _stdev = np.std(data_tmp[ifin])
                if err is not None:
                    err_tmp = err[tind]
                    _stdev += np.sqrt(np.sum(err_tmp[ifin] ** 2)) / len(ifin)
            else:
                _avrg = np.nan

        binned_data.append(_avrg)
        binned_grad.append(_grad)
        binned_err.append(_stdev)

        if debug:
            print(t, _avrg, _stdev)

    fin = np.isfinite(np.array(binned_data))
    if len(np.where(fin == True)[0]) > 2:
        binned_grad = np.gradient(np.array(binned_data), time_new)

    return np.array(binned_data), np.array(binned_err), np.array(binned_grad)


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
