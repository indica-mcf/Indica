"""
...write documentation...
"""

from copy import deepcopy
import pickle
from scipy import constants

import hda.fac_profiles as fac
from hda.forward_models import Spectrometer
import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray
from xarray import Dataset
import os
import json

from indica.readers import ST40Reader
from indica.readers import ADASReader

plt.ion()


def additional_info():
    info = {
        "te0": {"max": False, "label": "T$_e$(0)", "units": "(keV)", "const": 0.001},
        "ti0": {"max": False, "label": "T$_i$(0)", "units": "(keV)", "const": 0.001},
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


def analyse_database(regr_data):
    """
    Apply general filters to data and calculate maximum values of binned data
    """

    # Multiplication factor for calculation of Te,i(0)
    regr_data.temp_ratio = simulate_xrcs()

    # Apply general filters
    regr_data.binned = general_filters(regr_data.binned)

    # Calculate additional quantities and filter again
    regr_data.binned, regr_data.max_val, regr_data.info = calc_additional_quantities(
        regr_data.binned, regr_data.max_val, regr_data.info, regr_data.temp_ratio
    )
    regr_data.binned = general_filters(regr_data.binned)

    # Calculate max values of binned data
    regr_data.max_val = calc_max_val(
        regr_data.binned, regr_data.max_val, regr_data.info, t_max=regr_data.t_max
    )

    # Apply general filters and calculate additional quantities
    regr_data.max_val = general_filters(regr_data.max_val)

    # Apply defult selection criteria
    regr_data.filtered = apply_selection(regr_data.binned)

    return regr_data


def calc_additional_quantities(binned, max_val, info, temp_ratio):
    keys = list(info)

    empty_max_val = xr.full_like(max_val[keys[0]], np.nan)
    empty_binned = xr.full_like(binned[keys[0]], np.nan)

    # Estimate central temperature from parameterization
    binned = calc_central_temperature(binned, temp_ratio)

    pulses = binned["ipla_efit"].pulse
    time = binned["ipla_efit"].t

    # NBI power
    # TODO: propagation of gradient of V and I...
    info["nbi_power"] = {
        "max": False,
        "label": "P$_{NBI}$",
        "units": "(V * A)",
        "const": 1.0,
    }
    max_val["nbi_power"] = deepcopy(empty_max_val)
    max_val["nbi_power"].value.values = (
        max_val["i_hnbi"] * max_val["v_hnbi"]
    ).value.values

    binned["nbi_power"] = deepcopy(empty_binned)
    binned["nbi_power"].value.values = (
        binned["i_hnbi"] * binned["v_hnbi"]
    ).value.values
    binned["nbi_power"].error.values = np.sqrt(
        (binned["i_hnbi"].error.values * binned["v_hnbi"].value.values) ** 2
        + (binned["i_hnbi"].value.values * binned["v_hnbi"].error.values) ** 2
    )

    # Pulse length
    # Ip > 50 kA & up to end of flat-top
    info["pulse_length"] = {
        "max": False,
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
        "max": True,
        "label": "(R$_{geo}$ I$_P$ @ 10 ms) / (R$_{MC}$ max(I$_{MC})$)",
        "units": " ",
        "const": 1.0,
    }
    info["rip_imc"] = {
        "max": True,
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
        "max": True,
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
        "max": True,
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
        "max": True,
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
        "max": False,
        "label": "Cumulative NBI power",
        "units": "(kV * A * s)",
        "const": 1.0e-3,
    }
    max_val["total_nbi"] = deepcopy(empty_max_val)
    binned["total_nbi"] = deepcopy(empty_binned)
    binned["total_nbi"].value.values = binned["nbi_power"].cumul.values

    info["ti_te_xrcs"] = {
        "max": False,
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
        "max": False,
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

    return binned, max_val, info


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
            results[k] = xr.where(cond, results[k], 0,)

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
        "rip_imc",
    ]
    for k in neg_to_nan:
        if k in keys:
            cond = (results[k].value > 0) * (np.isfinite(results[k].value))
            results[k] = xr.where(cond, results[k], np.nan,)

    # Set to Nan if values outside specific ranges
    err_perc_keys = [
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
    err_perc_cond = {}
    for k in err_perc_keys:
        err_perc_cond[k] = {"var": "error", "lim": (np.nan, 0.2)}

    for k in err_perc_keys:
        if k in keys:
            cond = {k: err_perc_cond[k]}
            selection = selection_criteria(results, cond)
            results[k] = xr.where(selection, results[k], np.nan)

    lim_cond = {"wp_efit": {"var": "value", "lim": (np.nan, 100.0e3)}}
    for k in lim_cond:
        if k in keys:
            cond = {k: lim_cond[k]}
            selection = selection_criteria(results, cond)
            results[k] = xr.where(selection, results[k], np.nan)

    # Set to nan all the values where the gradients are nan
    if "t" in results[k].dims:
        grad_nan = [
            "te_xrcs",
            "ti_xrcs",
            "ti0",
            "te0",
            "ne_smmh1",
        ]
        for k in grad_nan:
            if k in keys:
                cond = np.isfinite(results[k].gradient)
                results[k].value.values = xr.where(
                    cond, results[k].value, np.nan,
                ).values

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
                    te_xrcs.values, temp_ratio[i].te_xrcs, temp_ratio[i].values,
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


def calc_max_val(binned, max_val, info, t_max=0.02, keys=None):
    """
    Calculate maximum value in a pulse using the binned data

    Parameters
    ----------
    t_max
        Time above which the max search should start

    """
    print("Calculating maximum values from binned data")

    # Calculate max values for those quantities where binned data is to be used
    if keys is None:
        keys = list(binned.keys())
    else:
        if type(keys) != list:
            keys = [keys]

    for k in keys:
        if k not in info.keys():
            print(f"\n Max val: key {k} not in info dictionary...")
            continue

        for p in binned[keys[0]].pulse:
            v = info[k]
            if v["max"] is True:
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


def write_to_pickle(regr_data):
    picklefile = f"{regr_data.path_data}{regr_data.pkl_file_data}"
    print(f"Saving regression database to \n {picklefile}")
    pickle.dump(regr_data, open(picklefile, "wb"))


def add_to_plot(xlab, ylab, tit, legend=True, vlines=False):
    if vlines:
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    if legend:
        plt.legend()


def set_data_info():
    # TODO: superseeded by info_json method
    """
    Info dictionary defining the data to read

    uid, diag, node, seq
        Information to read data from MDS+
    err
        None if no error node available, otherwise, path to error
    max
        True if already calculated for unbinned values
        False if this has to be calculated from binned data
    label
        label for plotting
    units
        unit of plot
    const
        constant to rescale units for plot
    """

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
            "const": 1e-06,
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
            "const": 0.001,
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
        "volm_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:volm",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "V$_p$ EFIT",
            "units": "",
            "const": 1.0,
        },
        "elon_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:elon",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "Elongation EFIT",
            "units": "",
            "const": 1.0,
        },
        "zmag_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:zmag",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "R$_{mag}$ EFIT",
            "units": "",
            "const": 1.0,
        },
        "rmag_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:rmag",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "R$_{mag}$ EFIT",
            "units": "",
            "const": 1.0,
        },
        "rmin_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:cr0",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "R$_{min}$ EFIT",
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
            "const": 1e-06,
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
            "const": 0.001,
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
            "const": 1e-19,
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
            "const": 1e-19,
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
            "const": 1000.0,
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
            "const": 0.001,
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
            "const": 0.001,
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
            "const": 0.001,
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
            "const": 0.001,
        },
        "i_hnbi": {
            "uid": "raw_nbi",
            "diag": "hnbi1",
            "node": ".hv_ps:i_jema",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "I$_{HNBI}$",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "v_hnbi": {
            "uid": "raw_nbi",
            "diag": "hnbi1",
            "node": ".hv_ps:v_jema",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "V$_{HNBI}$",
            "units": "(a.u.)",
            "const": 0.001,
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
        "i_bvl": {
            "uid": "",
            "diag": "psu",
            "node": ".bvl:i",
            "seq": -1,
            "err": None,
            "max": True,
            "sign": -1,
            "label": "I$_{BVL}$ PSU",
            "units": "(kA)",
            "const": 0.001,
        },
        "te0": {"max": False, "label": "T$_e$(0)", "units": "(keV)", "const": 0.001},
        "ti0": {"max": False, "label": "T$_i$(0)", "units": "(keV)", "const": 0.001},
        "d_i_6561": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.d_i_6561:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "D I 656.1 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "sum_ar": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".key_species:sum_ar",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "Ar lines",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "sum_b": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".key_species:sum_b",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "B lines",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "sum_c": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".key_species:sum_c",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "C lines",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "sum_h": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".key_species:sum_h",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "H lines",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "sum_he": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".key_species:sum_he",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "He lines",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "sum_li": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".key_species:sum_li",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "Li lines",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "sum_n": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".key_species:sum_n",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "N lines",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "sum_o": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".key_species:sum_o",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "O lines",
            "units": "(a.u.)",
            "const": 1.0,
        },
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

    return info


def fix_things(regr_data, assign=True):
    binned = deepcopy(regr_data.binned)

    for k in binned.keys():
        binned[k].gradient.values = binned[k].value.differentiate("t", edge_order=2)

    if assign:
        regr_data.binned = binned

    return regr_data


def test(regr_data):
    # density fringe jump
    p = 9709
    k = "ne_smmh1"

    # strange electron temperature
    p = 9781
    k = "te_xrcs"

    # high bremsstrahlung
    p = 9413
    k = "brems_pi"

    data = regr_data.binned[k].sel(pulse=p)

    plt.figure()
    data.value.plot(marker="o")
    plt.figure()
    data.gradient.plot(marker="o")

    # high bremsstrahlung
    plt.figure()
    data = regr_data.binned["sum_c"] / regr_data.binned["ne_smmh1"]
    data.value.sel(t=0.03, method="nearest").plot()

    plt.figure()
    data = regr_data.binned["sum_c"] / regr_data.binned["ne_nirh1"]
    data.value.sel(t=0.03, method="nearest").plot()

    plt.figure()
    data = regr_data.binned["brems_mp"] / regr_data.binned["ne_nirh1"]
    data.value.sel(t=0.03, method="nearest").plot()
