"""
...write documentation...
"""

from copy import deepcopy
from scipy import constants

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pylab as plt

from trends.trends_database import Database

plt.ion()


def additional_info():
    info = {
        "nbi_power": {
            "max": False,
            "label": "P$_{NBI}$",
            "units": "(V * A)",
            "const": 1.0,
        },
        "pulse_length": {
            "max": False,
            "label": "Pulse length",
            "units": "(s)",
            "const": 1.0,
        },
        "rip_efit": {
            "max": True,
            "label": "(R$_{geo}$ I$_P$ @ 10 ms) / (R$_{MC}$ max(I$_{MC})$)",
            "units": " ",
            "const": 1.0,
        },
        "rip_imc": {
            "max": True,
            "label": "(EFIT R$_{geo}$ I$_P$ @ 10 ms) / (R$_{MC}$ max(I$_{MC})$)",
            "units": " ",
            "const": 1.0,
        },
        "gas_cumulative": {
            "max": True,
            "label": "Cumulative gas",
            "units": "(V * s)",
            "const": 1.0,
        },
        "gas_prefill": {
            "max": True,
            "label": "Gas prefill",
            "units": "(V * s)",
            "const": 1.0,
        },
        "gas_fuelling": {
            "max": True,
            "label": "Gas fuelling t > 0",
            "units": "(V * s)",
            "const": 1.0,
        },
        "total_nbi": {
            "max": False,
            "label": "Cumulative NBI power",
            "units": "(kV * A * s)",
            "const": 0.001,
        },
        "ti_te_xrcs": {
            "max": False,
            "label": "Ti/Te (XRCS)",
            "units": "",
            "const": 1.0,
        },
        "ne_nirh1_te_xrcs": {
            "max": False,
            "label": "Ne (NIRH1)/3. * Te (XRCS)",
            "units": "",
            "const": 1.0,
        },
    }
    #
    # "te0": {"max": False, "label": "T$_e$(0)", "units": "(keV)", "const": 0.001},
    # "ti0": {"max": False, "label": "T$_i$(0)", "units": "(keV)", "const": 0.001},
    return info


def analyse_database(database):
    """
    Apply general filters to data and calculate maximum values of binned data
    """

    # Calculate additional quantities
    database = calc_additional_quantities(database)

    # Apply general filters to binned values and calculate its max value
    database.binned = general_filters(database.binned)

    # Calculate max values of binned data
    database.binned_max_val = calc_max_val(
        database, t_max=database.t_max
    )

    # Apply general filters and calculate additional quantities
    database.binned_max_val = general_filters(database.binned_max_val)

    # Apply defult selection criteria
    database.filtered = apply_selection(database.binned)

    return database


def calc_additional_quantities(database:Database):
    # Estimate central temperature from parameterization
    # database = calc_central_temperature(database)

    info = database.info
    keys = list(database.binned)
    binned = database.binned
    max_val = database.max_val
    empty_binned = xr.full_like(binned[keys[0]], np.nan)
    empty_max_val = xr.full_like(max_val[keys[0]], np.nan)

    pulses = database.binned["ipla_efit"].pulse
    time = database.binned["ipla_efit"].t

    # NBI power
    # TODO: propagation of gradient of V and I...
    info["nbi_power"] = {
        "label": "P$_{NBI}$",
        "units": "(V * A)",
        "const": 1.0,
    }
    binned["nbi_power"] = empty_binned
    binned["nbi_power"].value.values = (
        binned["i_hnbi"] * binned["v_hnbi"]
    ).value.values
    binned["nbi_power"].error.values = np.sqrt(
        (binned["i_hnbi"].error.values * binned["v_hnbi"].value.values) ** 2
        + (binned["i_hnbi"].value.values * binned["v_hnbi"].error.values) ** 2
    )

    max_val["nbi_power"] = deepcopy(empty_max_val)
    max_val["nbi_power"].value.values = (
        max_val["i_hnbi"] * max_val["v_hnbi"]
    ).value.values

    # Pulse length
    # Ip > 50 kA & up to end of flat-top
    info["pulse_length"] = {
        "label": "Pulse length",
        "units": "(s)",
        "const": 1.0,
    }

    cond = {
        "Flattop": {
            "ipla_efit": {"var": "value", "lim": (50.0e3, np.nan)},
            "ipla_efit": {"var": "gradient", "lim": (-1e6, np.nan)},
        }
    }
    filtered = apply_selection(binned, cond, default=False)

    pulse_length = []
    for pulse in pulses:
        tind = np.where(filtered["Flattop"]["selection"].sel(pulse=pulse) == True)[0]
        if len(tind) > 0:
            pulse_length.append(time[tind.max()])
        else:
            pulse_length.append(0)
    max_val["pulse_length"] = deepcopy(empty_max_val)
    max_val["pulse_length"].value.values = np.array(pulse_length)

    # Calculate RIP/IMC and add to values to be plotted
    info["rip_efit"] = {
        "label": "(R$_{geo}$ I$_P$ @ 10 ms) / (R$_{MC}$ max(I$_{MC})$)",
        "units": " ",
        "const": 1.0,
    }
    info["rip_imc"] = {
        "label": "(EFIT R$_{geo}$ I$_P$ @ 10 ms) / (R$_{MC}$ max(I$_{MC})$)",
        "units": " ",
        "const": 1.0,
    }
    binned["rip_efit"] = deepcopy(empty_binned)
    binned["rip_efit"].value.values = (
        binned["rmag_efit"].value * binned["ipla_efit"].value.values
    )

    rip_efit = binned["rip_efit"].sel(t=0.015, method="nearest").drop("t")
    max_val["rip_imc"] = deepcopy(empty_max_val)
    max_val["rip_imc"] = rip_efit / (max_val["imc"] * 0.75 * 22)
    max_val["rip_imc"].error.values = rip_efit.error.values / (
        max_val["imc"].value.values * 0.75 * 22
    )
    # Calculate total gas puff = cumulative gas_puff & its max value
    info["gas_cumulative"] = {
        "label": "Cumulative gas",
        "units": "(V * s)",
        "const": 1.0,
    }
    binned["gas_cumulative"] = deepcopy(empty_binned)
    binned["gas_cumulative"].value.values = binned["gas_puff"].cumul.values

    max_val["gas_cumulative"] = deepcopy(empty_max_val)
    pulse_length = xr.where(
        max_val["pulse_length"].value > 0.04, max_val["pulse_length"].value, np.nan
    )
    max_val["gas_cumulative"].value.values = (
        binned["gas_cumulative"].value.max("t").values / pulse_length.values
    )

    # Gas prefill
    info["gas_prefill"] = {
        "label": "Gas prefill",
        "units": "(V * s)",
        "const": 1.0,
    }
    max_val["gas_prefill"] = deepcopy(empty_max_val)
    max_val["gas_prefill"].value.values = (
        binned["gas_puff"].cumul.sel(t=0, method="nearest").values
    )

    # Gas for t > 0
    info["gas_fuelling"] = {
        "label": "Gas fuelling t > 0",
        "units": "(V * s)",
        "const": 1.0,
    }
    max_val["gas_fuelling"] = deepcopy(empty_max_val)
    max_val["gas_fuelling"].value.values = (
        binned["gas_cumulative"].cumul.max("t")
        - binned["gas_puff"].cumul.sel(t=0, method="nearest")
    ).values

    # Calculate total gas puff = cumulative gas_puff & its max value
    info["total_nbi"] = {
        "label": "Cumulative NBI power",
        "units": "(kV * A * s)",
        "const": 1.0e-3,
    }
    max_val["total_nbi"] = deepcopy(empty_max_val)
    binned["total_nbi"] = deepcopy(empty_binned)
    binned["total_nbi"].value.values = binned["nbi_power"].cumul.values

    info["ti_te_xrcs"] = {
        "label": "Ti/Te (XRCS)",
        "units": "",
        "const": 1.0,
    }
    max_val["ti_te_xrcs"] = deepcopy(empty_max_val)
    binned["ti_te_xrcs"] = deepcopy(empty_binned)
    binned["ti_te_xrcs"].value.values = (
        binned["ti_xrcs"].value.values / binned["te_xrcs"].value.values
    )
    binned["ti_te_xrcs"].error.values = binned["ti_te_xrcs"].value.values * np.sqrt(
        (binned["ti_xrcs"].error.values / binned["ti_xrcs"].value.values) ** 2
        + (binned["ti_xrcs"].error.values / binned["ti_xrcs"].value.values) ** 2
    )

    info["ne_nirh1_te_xrcs"] = {
        "label": "Ne (NIRH1)/3. * Te (XRCS)",
        "units": "",
        "const": 1.0,
    }
    max_val["ne_nirh1_te_xrcs"] = deepcopy(empty_max_val)
    binned["ne_nirh1_te_xrcs"] = deepcopy(empty_binned)
    binned["ne_nirh1_te_xrcs"].value.values = (
        binned["ne_nirh1"].value.values
        / 3.0
        * binned["te_xrcs"].value.values
        * constants.e
    )
    binned["ne_nirh1_te_xrcs"].error.values = binned[
        "ne_nirh1_te_xrcs"
    ].value.values * np.sqrt(
        (binned["ne_nirh1"].error.values / binned["ne_nirh1"].value.values) ** 2
        + (binned["te_xrcs"].error.values / binned["te_xrcs"].value.values) ** 2
    )

    database.empty_max_val = empty_max_val
    database.empty_binned = empty_binned
    database.binned = binned
    database.max_val = max_val

    return database


def general_filters(results):
    """
    Apply general filters to data read e.g. NBI power 0 where not positive

    TODO: add detection of fringe jumps for SMMH1 --> using gradient?
    """
    print("Applying general data filters")
    keys = results.keys()

    # Set all negative values to 0
    neg_to_zero = ["nbi_power"]
    for k in neg_to_zero:
        if k in keys:
            cond = (results[k].value > 0) * np.isfinite(results[k].value)
            dims = results[k].value.dims
            for ds_key in list(results[k]):
                if results[k][ds_key].dims == dims:
                    results[k][ds_key].values = xr.where(cond, results[k][ds_key], 0,).values

    # Set all negative values to Nan
    neg_to_nan = [
        "te_xrcs",
        "ti_xrcs",
        "ipla_efit",
        "ipla_pfit",
        "wp_efit",
        "ne_nirh1",
        "ne_smmh1",
        "gas_press",
        "rip_imc",
    ]
    #
    # "ti0",
    # "te0",
    for k in neg_to_nan:
        if k in keys:
            cond = (results[k].value > 0) * (np.isfinite(results[k].value))
            dims = results[k].value.dims
            for ds_key in list(results[k]):
                if results[k][ds_key].dims == dims:
                    results[k][ds_key].values = xr.where(cond, results[k][ds_key], np.nan,).values

    # Set to Nan if values outside specific ranges
    err_perc_keys = [
        "te_xrcs",
        "ti_xrcs",
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
    #
    # "te0",
    # "ti0",
    err_perc_cond = {}
    for k in err_perc_keys:
        err_perc_cond[k] = {"var": "error", "lim": (np.nan, 0.2)}

    for k in err_perc_keys:
        if k in keys:
            cond = {k: err_perc_cond[k]}
            selection = selection_criteria(results, cond)
            dims = results[k].value.dims
            for ds_key in list(results[k]):
                if results[k][ds_key].dims == dims:
                    results[k][ds_key].values = xr.where(selection, results[k][ds_key], np.nan,).values

    lim_cond = {"wp_efit": {"var": "value", "lim": (np.nan, 100.0e3)}}
    for k in lim_cond:
        if k in keys:
            cond = {k: lim_cond[k]}
            selection = selection_criteria(results, cond)
            dims = results[k].value.dims
            for ds_key in list(results[k]):
                if results[k][ds_key].dims == dims:
                    results[k][ds_key].values = xr.where(selection, results[k][ds_key], np.nan,).values

    # Set to nan all the values where the gradients are nan
    if "t" in results[k].dims:
        grad_nan = [
            "te_xrcs",
            "ti_xrcs",
            "ne_smmh1",
        ]
        #
        # "ti0",
        # "te0",
        for k in grad_nan:
            if k in keys:
                cond = np.isfinite(results[k].gradient)
                dims = results[k].value.dims
                for ds_key in list(results[k]):
                    if results[k][ds_key].dims == dims:
                        results[k][ds_key].values = xr.where(cond, results[k][ds_key], np.nan, ).values

    return results


def calc_central_temperature(database:Database):
    print("Calculating central temperature from parameterization")

    # Central temperatures from XRCS parametrization
    temp_ratio = simulate_xrcs()

    mult_binned = []
    profs = np.arange(len(temp_ratio))
    for i in range(len(temp_ratio)):
        ratio_tmp = xr.full_like(database.binned["te_xrcs"].value, np.nan)
        # TODO: DataArray interp crashing if all nans (@ home only)
        for p in database.binned["te_xrcs"].pulse:
            te_xrcs = database.binned["te_xrcs"].value.sel(pulse=p)
            if any(np.isfinite(te_xrcs)):
                ratio_tmp.loc[dict(pulse=p)] = np.interp(
                    te_xrcs.values, temp_ratio[i].te_xrcs, temp_ratio[i].values,
                )
        mult_binned.append(ratio_tmp)
    mult_binned = xr.concat(mult_binned, "prof").assign_coords({"prof": profs})

    # Binned data
    mult_max = mult_binned.max("prof", skipna=True)
    mult_min = mult_binned.min("prof", skipna=True)
    mult_mean = mult_binned.mean("prof", skipna=True)
    print(mult_mean)
    database.binned["te0"].value.values = (database.binned["te_xrcs"].value * mult_mean).values
    err = np.abs(database.binned["te0"].value * mult_max - database.binned["te0"].value * mult_min)
    database.binned["te0"].error.values = np.sqrt(
        (database.binned["te_xrcs"].error * mult_mean) ** 2 + err ** 2
    ).values
    database.binned["ti0"].value.values = (database.binned["ti_xrcs"].value * mult_mean).values
    err = np.abs(database.binned["ti0"].value * mult_max - database.binned["ti0"].value * mult_min)
    database.binned["ti0"].error.values = np.sqrt(
        (database.binned["ti_xrcs"].error * mult_mean) ** 2 + err ** 2
    ).values

    return database


def calc_max_val(database:Database, t_max=0.02, keys=None):
    """
    Calculate maximum value in a pulse using the binned data

    Parameters
    ----------
    t_max
        Time above which the max search should start

    """
    print("Calculating maximum values from binned data")

    binned = database.binned
    binned_max_val = deepcopy(database.max_val)
    info = database.info

    # Calculate max values for those quantities where binned data is to be used
    if keys is None:
        keys = list(binned.keys())
    else:
        if type(keys) != list:
            keys = [keys]

    keys = list(binned)
    empty_max_val = xr.full_like(binned_max_val[keys[0]], np.nan)

    for k in keys:
        if k not in info.keys():
            print(f"\n Max val: key {k} not in info dictionary...")
            continue

        if k not in binned_max_val.keys():
            binned_max_val[k] = empty_max_val

        for p in binned[keys[0]].pulse:
            max_search = xr.where(
                binned[k].t > t_max, binned[k].value.sel(pulse=p), np.nan
            )
            if not any(np.isfinite(max_search)):
                binned_max_val[k].value.loc[dict(pulse=p)] = np.nan
                binned_max_val[k].error.loc[dict(pulse=p)] = np.nan
                binned_max_val[k].time.loc[dict(pulse=p)] = np.nan
                continue
            tind = max_search.argmax(dim="t", skipna=True).values
            tmax = binned[k].t[tind]
            binned_max_val[k].time.loc[dict(pulse=p)] = tmax
            binned_max_val[k].value.loc[dict(pulse=p)] = binned[k].value.sel(pulse=p, t=tmax)
            binned_max_val[k].error.loc[dict(pulse=p)] = binned[k].error.sel(pulse=p, t=tmax)

    return binned_max_val


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


def write_to_csv(database:Database):

    results = database.filtered
    ipla_max = results["ipla_efit_max"].value.values
    ipla_max_time = results["ipla_efit_max"].time.values
    ti_max = results["ti_xrcs_max"].value.values
    ti_max_err = results["ti_xrcs_max"].error.values
    ti_max_time = results["ti_xrcs_max"].time.values
    ratio = database.temp_ratio.sel(
        te_xrcs=results["te_xrcs_max"].value.values, method="nearest"
    ).values
    ti_0 = ti_max * ratio
    ipla_at_max = []
    nbi_at_max = []

    pulses = database.pulses
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


def add_to_plot(xlab, ylab, tit, legend=True, vlines=False):
    if vlines:
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    if legend:
        plt.legend()

