"""
...write documentation...
"""

from copy import deepcopy
from scipy import constants
from xarray import DataArray, Dataset
from matplotlib import cm, rcParams
from MDSplus.mdsExceptions import TreeFOPENR

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pylab as plt

from trends.trends_database import Database
import hda.physics as ph

plt.ion()


def analyse_database(database: Database):
    """
    Apply general filters to data and calculate maximum values of binned data
    """

    # Apply general filters to binned values and calculate its max value
    database.binned = general_filters(database.binned)

    # Calculate additional quantities
    database = calc_additional_quantities(database)

    # Calculate max values of binned data
    database.binned_max_val, _ = calc_max_val(database.binned, t_max=database.t_max)

    # Apply general filters on all values including additional quantities
    database.binned_max_val = general_filters(database.binned_max_val)

    # Apply defult selection criteria
    database.filtered = apply_selection(database.binned)

    return database


def calc_additional_quantities(database: Database):
    info = database.info
    binned = database.binned
    max_val = database.max_val

    k = list(binned.keys())[0]
    empty_binned = xr.full_like(database.binned[k], np.nan)
    empty_max_val = xr.full_like(database.max_val[k], np.nan)
    pulses = database.binned[k].pulse
    time = database.binned[k].t

    # Flat-top pulse length
    info["pulse_length"] = {
        "label": "Pulse length",
        "units": "(s)",
        "const": 1.0,
    }
    cond = {
        "Flattop": {
            "ipla_efit:value": (50.0e3, np.nan),
            "ipla_efit:gradient": (-1e6, np.nan),
        }
    }
    filtered = apply_selection(binned, cond)
    pulse_length = []
    for pulse in pulses:
        tind = np.where(filtered["Flattop"]["selection"].sel(pulse=pulse) == True)[0]
        if len(tind) > 0:
            pulse_length.append(time[tind.max()])
        else:
            pulse_length.append(0)
    max_val["pulse_length"] = deepcopy(empty_max_val)
    max_val["pulse_length"].value.values = np.array(pulse_length)

    # Ohmic power from loop voltage and plasma current
    info["vloop_ipla"] = {
        "label": "Ohmic power V$_{loop}$ x I$_P$",
        "units": "(MW)",
        "const": 1.0e-6,
    }
    binned["vloop_ipla"] = deepcopy(empty_binned)
    max_val["vloop_ipla"] = deepcopy(empty_max_val)
    binned["vloop_ipla"].value.values = (
        binned["ipla_efit"].value * binned["vloop_l016"].value
    ).values
    # binned["vloop_ipla"].error.values = (binned["vloop_ipla"].value).values

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
    info["nbi_power"] = {
        "label": "Cumulative NBI power",
        "units": "(kW)",
        "const": 1.0e-3,
    }
    binned["nbi_power"] = xr.full_like(empty_binned, 0.0)
    max_val["nbi_power"] = xr.full_like(empty_max_val, 0.0)
    binned["nbi_power"].value.values = (
        binned["p_hnbi1"].value + binned["p_rfx"].value
    ).values
    binned["nbi_power"].error.values = (
        binned["p_hnbi1"].error + binned["p_rfx"].error
    ).values
    binned["nbi_power"].cumul.values = (
        binned["p_hnbi1"].cumul + binned["p_rfx"].cumul
    ).values

    info["hnbi1_on"] = {
        "label": "Beam on (HNBI1)",
        "units": "",
        "const": 1.0,
    }
    hnbi1 = database.binned["p_hnbi1"].value > 0.1
    hnbi1.values = np.int_(hnbi1)
    binned["hnbi1_on"] = deepcopy(empty_binned)
    binned["hnbi1_on"].value.values = hnbi1.values
    t_hnbi = hnbi1.cumsum("t") * database.dt * database.overlap
    binned["hnbi1_on"].cumul.values = t_hnbi.values
    max_val["hnbi1_on"] = deepcopy(empty_max_val)

    info["rfx_on"] = {
        "label": "Beam on (RFX)",
        "units": "",
        "const": 1.0,
    }
    rfx = database.binned["p_rfx"].value > 0.1
    rfx.values = np.int_(rfx)
    binned["rfx_on"] = deepcopy(empty_binned)
    binned["rfx_on"].value.values = rfx.values
    t_rfx = rfx.cumsum("t") * database.dt * database.overlap * rfx
    binned["rfx_on"].cumul.values = t_rfx.values
    binned["rfx_on"].value.values = xr.where(
        binned["rfx_on"].value > 0, binned["rfx_on"].value, np.nan
    ).values
    max_val["rfx_on"] = deepcopy(empty_max_val)

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

    # database.empty_max_val = empty_max_val
    # database.empty_binned = empty_binned
    # database.binned = binned
    # database.max_val = max_val
    # database.info = info

    return database


def general_filters(results: dict):
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
                    results[k][ds_key].values = xr.where(
                        cond, results[k][ds_key], 0,
                    ).values

    # Set all negative values to Nan
    neg_to_nan = [
        "te_xrcs",
        "ti_xrcs",
        "ipla_efit",
        "ipla_pfit",
        "wp_efit",
        "ne_nirh1",
        "ne_smmh1",
        "p_hnbi1",
        "p_rfx",
        "gas_press",
        "rip_imc",
        "ne_nirh1_te_xrcs",
    ]
    for k in neg_to_nan:
        if k in keys:
            cond = (results[k].value > 0) * (np.isfinite(results[k].value))
            dims = results[k].value.dims
            for ds_key in list(results[k]):
                if results[k][ds_key].dims == dims:
                    results[k][ds_key].values = xr.where(
                        cond, results[k][ds_key], np.nan,
                    ).values

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
    ]
    err_perc_cond = {}
    for k in err_perc_keys:
        err_perc_cond[k] = {f"{k}:error": (np.nan, 0.2)}

    for k, cond in err_perc_cond.items():
        if k in keys:
            selection = selection_criteria(results, cond)
            dims = results[k].value.dims
            for ds_key in list(results[k]):
                if results[k][ds_key].dims == dims:
                    results[k][ds_key].values = xr.where(
                        selection, results[k][ds_key], np.nan,
                    ).values

    lim_cond = {"wp_efit": {"wp_efit:value": (0, 100.0e3)}}
    for k, cond in lim_cond.items():
        if k in keys:
            selection = selection_criteria(results, cond)
            dims = results[k].value.dims
            for ds_key in list(results[k]):
                if results[k][ds_key].dims == dims:
                    results[k][ds_key].values = xr.where(
                        selection, results[k][ds_key], np.nan,
                    ).values

    # Set to nan all the values where the gradients are nan
    if "t" in results[k].dims:
        grad_nan = [
            "te_xrcs",
            "ti_xrcs",
            "ne_smmh1",
        ]
        for k in grad_nan:
            if k in keys:
                cond = np.isfinite(results[k].gradient)
                dims = results[k].value.dims
                for ds_key in list(results[k]):
                    if results[k][ds_key].dims == dims:
                        results[k][ds_key].values = xr.where(
                            cond, results[k][ds_key], np.nan,
                        ).values

    return results


def calc_max_val(binned: dict, t_max=0.02, keys=None):
    """
    Calculate maximum value in a pulse using the binned data

    Parameters
    ----------
    t_max
        Time above which the max search should start

    """
    print("Calculating maximum values from binned data")

    if keys is None:
        keys = list(binned)
    else:
        if type(keys) != list:
            keys = [keys]

    empty = xr.full_like(
        binned[keys[0]].interp(t=0, method="nearest").drop("t"), np.nan
    )
    empty["time"] = deepcopy(empty.value)

    tslice = slice(t_max, float(binned[keys[0]].t.max()))
    max_val, min_val = {}, {}
    for k in keys:
        max_val[k] = deepcopy(empty)
        min_val[k] = deepcopy(empty)

        _search = binned[k].value.sel(t=tslice)

        tind = _search.fillna(-1.0e30).argmax(dim="t")
        tmax = _search.t.isel(t=tind).drop("t")
        max_val[k].time.values = tmax
        max_val[k].value.values = binned[k].value.sel(t=tmax).values
        max_val[k].error.values = binned[k].error.sel(t=tmax).values

        tind = _search.fillna(+1.0e30).argmin(dim="t")
        tmin = _search.t.isel(t=tind).drop("t")
        min_val[k].time.values = tmin
        min_val[k].value.values = binned[k].value.sel(t=tmin).values
        min_val[k].error.values = binned[k].error.sel(t=tmin).values

    return max_val, min_val


def selection_criteria(binned: dict, cond: dict):
    """
    Find values within specified limits

    Parameters
    ----------
    binned
        Database binned result dictionary
    cond
        Dictionary of database keys with respective limits e.g.
        {"identifier:variable":(lower_limit, upper_limit)} =
        {"ne_nirh1:value":(0, 2.e19)}
        where:
        - "identifier" = variable_diagnostic to e.g. "ne_nirh1" is the key of results dictionary
        - "variable" = dataset variable to be used for the selection,
        either "value", "perc_error", "gradient", "norm_gradient"

    Returns
    -------
        Boolean Dataarray of the same shape as the binned data with
        items == True if satisfying the selection criteria

    """

    k = list(binned.keys())[0]

    selection = xr.where(xr.ones_like(binned[k].value) == 1, True, False)
    for c, lim in cond.items():
        identifier, variable = c.split(":")

        item = binned[identifier]
        if variable == "error":  # percentage error
            val = np.abs(item["error"] / item["value"])
        else:
            val = item[variable]

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
    binned: dict, cond: bool = None,
):
    """
    Apply selection criteria as defined in the cond dictionary

    Parameters
    ----------
    binned
        Database class result dictionary of binned quantities
    cond
        Dictionary of selection criteria
        Different elements in list give different selection, elements
        in sub-dictionary are applied together (&)

    Returns
    -------

    """
    # TODO: max_val calculation too time-consuming...is it worth it?
    if cond is None:
        cond = {
            "NBI": {"nbi_power:value": (20, np.nan),},
            "Ohmic": {"nbi_power:value": (0,),},
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


def write_to_csv(database: Database):

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


def add_to_plot(
    xlab: str, ylab: str, tit: str, legend: bool = True, vlines: bool = False
):
    if vlines:
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    if legend:
        plt.legend()


def absorbed_nbi_power(
    database: Database = None,
    data=None,
    check_tevol=False,
    check_vloop=False,
    check_pulses=False,
    savefig=False,
):
    from indica.readers import ST40Reader

    def save_figure(fig_name="", orientation="landscape", ext=".jpg"):
        plt.savefig(
            "/home/marco.sertoli/figures/regr_trends/" + fig_name + ext,
            orientation=orientation,
            dpi=600,
            pil_kwargs={"quality": 95},
        )

    if database is None:
        database = Database(reload=True)

    binned = database.binned

    t0 = 0.01
    t1 = 0.15

    params = {
        9766: {"tlim": (0.04, 0.1)},
        9779: {"tlim": (0.04, 0.1)},
        9780: {"tlim": (0.04, 0.1)},
        9781: {"tlim": (0.04, 0.1)},
        9831: {"tlim": (0.04, 0.08)},
        9835: {"tlim": (0.04, 0.12)},
        9837: {"tlim": (0.04, 0.12)},
        9839: {"tlim": (0.04, 0.12)},
        9849: {"tlim": (0.04, 0.12)},
        9892: {"tlim": (0.04, 0.12)},
        # 10013: {"tlim": (0.04, 0.08)},
        # 10014: {"tlim": (0.04, 0.08)},
    }
    revs = np.arange(64, 76 + 1, 1)
    peaking = np.array(["broad"] * len(np.where(revs <= 68)[0]))
    peaking = np.append(peaking, np.array(["peaked"] * len(np.where(revs > 68)[0])))
    peaking = list(peaking)

    if data is None:
        data = {}
        for pulse in params.keys():
            reader = ST40Reader(pulse, t0, t1, tree="ST40")
            efit = reader.get("", "efit", 0)

            # Selection criteria for which NBI is on
            hnbi1 = binned["p_hnbi1"].sel(pulse=pulse).value > 0.1
            rfx = binned["p_rfx"].sel(pulse=pulse).value > 0.1
            _ti_xrcs = binned["ti_xrcs"].sel(pulse=pulse)
            _te_xrcs = binned["te_xrcs"].sel(pulse=pulse)

            _rfx_only = rfx * np.logical_not(hnbi1)
            _rfx_only.values = np.int_(_rfx_only)
            _hnbi1_only = hnbi1 * np.logical_not(rfx)
            _hnbi1_only.values = np.int_(_hnbi1_only)
            _both_beams = rfx * hnbi1
            _both_beams.values = np.int_(_both_beams)

            astra, fabs, rfx_only, hnbi1_only, both_beams, Ne_smmh1 = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            ti_xrcs, te_xrcs = [], []
            try:
                reader = ST40Reader(13100000 + pulse, t0, t1, tree="ASTRA")
                for rev in revs:
                    a = reader.get("", "astra", revision=rev)
                    astra.append(a)
                    fabs.append((a["pabs"] / a["pnb"]))

                    Ne = a["ne"] * 1.0e19
                    t = Ne.t
                    rho = Ne.rho_poloidal

                    rmjo = efit["rmjo"].interp(t=t).drop("z")
                    rmno = (rmjo - rmjo.min("rho_poloidal")).interp(rho_poloidal=rho)
                    rmno = xr.where(rmno >= 0, rmno, 0)
                    rmji = efit["rmji"].interp(t=t).drop("z")
                    rmni = (rmji - rmji.min("rho_poloidal")).interp(rho_poloidal=rho)
                    rmni = xr.where(rmni >= 0, rmni, 0)

                    _Ne_smmh1 = Ne.mean("rho_poloidal")
                    _Ne_smmh1.values = 2 * np.array(
                        [
                            np.trapz(Ne.sel(t=_t), rmno.sel(t=_t))
                            - np.trapz(Ne.sel(t=_t), rmni.sel(t=_t))
                            for _t in t
                        ]
                    )
                    Ne_smmh1.append(_Ne_smmh1)

                    rfx_only.append(_rfx_only.interp(t=t) == 1)
                    hnbi1_only.append(_hnbi1_only.interp(t=t) == 1)
                    both_beams.append(_both_beams.interp(t=t) == 1)
                    ti_xrcs.append(_ti_xrcs.interp(t=t))
                    te_xrcs.append(_te_xrcs.interp(t=t))

                data[pulse] = {
                    "astra": astra,
                    "fabs": fabs,
                    "Ne_smmh1": Ne_smmh1,
                    "rfx_only": rfx_only,
                    "hnbi1_only": hnbi1_only,
                    "both_beams": both_beams,
                    "ti_xrcs": ti_xrcs,
                    "te_xrcs": te_xrcs,
                    "rev": revs,
                    "peaking": peaking,
                }
            except TreeFOPENR:
                print(f"No ASTRA for pulse {pulse}")

    # Plot results
    const_ne = database.info["ne_smmh1"]["const"]
    label_ne = (
        f"{database.info['ne_smmh1']['label']} {database.info['ne_smmh1']['units']}"
    )
    label_t = "Time (s)"
    label_fabs = "NBI absorbed fraction"

    # Absorbed fraction vs. electron density
    plt.figure()
    for pulse, dd in data.items():
        tlim = params[pulse]["tlim"]
        _slice = slice(tlim[0], tlim[1])

        revs = dd["rev"]
        both = (
            xr.concat(dd["both_beams"], "rev")
            .assign_coords(rev=revs)
            .mean("rev", skipna=True)
        )
        Ne = (
            xr.concat(dd["Ne_smmh1"], "rev")
            .assign_coords(rev=revs)
            .mean("rev", skipna=True)
        )
        fabs = xr.concat(dd["fabs"], "rev").assign_coords(rev=revs, skipna=True)
        fabs = xr.where(fabs > 0.35, fabs, np.nan)
        fabs_mean = fabs.mean("rev", skipna=True)
        fabs_std = fabs.std("rev", skipna=True)

        x = Ne.sel(t=_slice).values * const_ne
        ylow = xr.where(both, (fabs_mean - fabs_std), np.nan).sel(t=_slice).values
        yhigh = xr.where(both, (fabs_mean + fabs_std), np.nan).sel(t=_slice).values
        ind = np.argsort(x)
        plt.fill_between(x[ind], ylow[ind], yhigh[ind], alpha=0.7, label=pulse)

    plt.title(label_fabs)
    plt.xlabel(label_ne)
    plt.ylabel("f$_{abs}$")
    plt.legend()
    plt.ylim(0, 1.2)
    plt.xlim(2, 6)
    if savefig:
        save_figure(fig_name="Pabs_vs_Ne")

    # Absorbed fraction vs. time since start of both beams (neglect HNBI start)
    plt.figure()
    for pulse, dd in data.items():
        tlim = params[pulse]["tlim"]
        _slice = slice(tlim[0], tlim[1])

        revs = dd["rev"]
        both = (
            xr.concat(dd["both_beams"], "rev")
            .assign_coords(rev=revs)
            .mean("rev", skipna=True)
        )
        fabs = xr.concat(dd["fabs"], "rev").assign_coords(rev=revs, skipna=True)
        fabs = xr.where(fabs > 0.35, fabs, np.nan)
        fabs_mean = fabs.mean("rev", skipna=True)
        fabs_std = fabs.std("rev", skipna=True)

        t_both = both.t[np.where(both == 1)[0][0]].values
        x = (fabs_mean.t - t_both).values
        ylow = xr.where(both, (fabs_mean - fabs_std), np.nan).values
        yhigh = xr.where(both, (fabs_mean + fabs_std), np.nan).values
        ind = np.where(x >= 0)[0]
        plt.fill_between(x[ind], ylow[ind], yhigh[ind], alpha=0.7, label=pulse)

    plt.title(label_fabs)
    plt.xlabel(label_t)
    plt.ylabel("f$_{abs}$")
    plt.legend()
    plt.ylim(0, 1.2)
    if savefig:
        save_figure(fig_name="Pabs_vs_NBI_start_time")

    for pulse, dd in data.items():
        print(pulse)
        tlim = params[pulse]["tlim"]
        _slice = slice(tlim[0], tlim[1])

        astra = dd["astra"]
        vloop_l016 = binned["vloop_l016"]
        vloop_l026 = binned["vloop_l026"]
        ipla = binned["ipla_efit"]

        fabs = xr.concat(dd["fabs"], "rev").assign_coords(rev=revs)
        _fabs = xr.where(fabs > 0.35, fabs, np.nan)
        fabs_mean = _fabs.mean("rev")
        fabs_std = _fabs.std("rev")

        if check_pulses:
            plt.figure()
            for i in range(len(revs)):
                f = dd["fabs"][i]
                rev = dd["rev"][i]
                p = dd["peaking"][i]
                Ne = dd["Ne_smmh1"][i]
                rfx = dd["rfx_only"][i]
                hnbi1 = dd["hnbi1_only"][i]
                both = dd["both_beams"][i]
                peaking = dd["peaking"][i]

                x = Ne.sel(t=_slice).values * const_ne

                marker = "o"
                y = xr.where(both, f, np.nan).sel(t=_slice).values
                plt.plot(x, y, marker=marker, linestyle="", label=f"{rev} {p}")

                marker = "+"
                y = xr.where(hnbi1, f, np.nan).sel(t=_slice).values
                plt.plot(x, y, marker=marker, linestyle="", markersize=12)

                marker = "x"
                y = xr.where(rfx, f, np.nan).sel(t=_slice).values
                plt.plot(x, y, marker=marker, linestyle="", markersize=12)

            ylow = (fabs_mean - fabs_std).sel(t=_slice).values
            yhigh = (fabs_mean + fabs_std).sel(t=_slice).values
            plt.fill_between(x, ylow, yhigh, alpha=0.7)
            plt.title(f"{pulse}")
            plt.xlabel(label_ne)
            plt.ylabel(label_fabs)
            plt.legend()
            plt.ylim(0, 1.2)
            plt.xlim(2, 6)
            if savefig:
                save_figure(fig_name=f"{pulse}_Pabs_vs_Ne")

        # Check Ohmic power (V * I) vs ASTRA P_OH
        if check_vloop:
            plt.figure()
            const = 1.0e-6
            for i in range(len(revs)):
                p_oh = astra[i]["p_oh"] * const
                rev = dd["rev"][i]
                p = dd["peaking"][i]
                peaking = dd["peaking"][i]
                p_oh.plot(marker="o", linestyle="", label=f"{rev} {peaking}")

            tmp = (vloop_l016 * ipla).sel(pulse=pulse, t=_slice).value * const
            tmp.plot(label="vloop_l016", color="black", linewidth=4)

            tmp = (vloop_l026 * ipla).sel(pulse=pulse, t=_slice).value * const
            tmp.plot(label="vloop_l026", color="black", linewidth=4, linestyle="dotted")

            plt.title(f"Pulse {pulse}")
            plt.ylabel("V$_{loop}$ * I$_P$ (MW)")
            plt.legend()
            if savefig:
                save_figure(fig_name=f"{pulse}_IxV_vs_Poh_ASTRA")

        if check_tevol:
            to_plot = [
                "ipla_efit",
                "ne_smmh1",
                "volm_efit",
                "rmag_efit",
                "p_hnbi1",
                "p_rfx",
            ]
            for k in to_plot:
                print(k)
                plt.figure()
                for pulse, dd in data.items():
                    const = database.info[k]["const"]
                    tmp = binned[k].sel(pulse=pulse, t=_slice).value * const
                    tmp.plot(label=f"{pulse}", alpha=0.8)
                plt.title(database.info[k]["label"])
                plt.ylabel(database.info[k]["units"])
                plt.legend()

    return database, data
    #
    # from scipy.interpolate import CubicSpline
    #
    # _Ne = [1.0e19, 3.0e19, 6.0e19, 1.0e20]
    # _fabs_hnbi1 = DataArray([0.4, 0.6, 0.8, 1.0], coords=[("electron_density", _Ne)],)
    # _fabs_rfx = DataArray([0.5, 0.7, 0.9, 1.0], coords=[("electron_density", _Ne)],)
    # _fabs_hnbi1.plot()
    # _fabs_rfx.plot()
    #
    # cubicspline = CubicSpline(
    #     _fabs.electron_density, _fabs.values, bc_type="natural", extrapolate=False,
    # )
    #
    # Ne = np.linspace(1.0e19, 2.0e20)
    # fabs = DataArray(cubicspline(Ne), coords=[("electron_density", Ne)])
    # fabs = xr.where(
    #     fabs.electron_density < _fabs.electron_density.min(), _fabs.min(), fabs,
    # )
    # fabs = xr.where(
    #     fabs.electron_density > _fabs.electron_density.max(), _fabs.max(), fabs,
    # )
    #
    # p_abs = binned["p_hnbi1"]


def plot_max_ti(save_data=False):
    import hda.physics as ph
    import datetime
    import os
    import json

    path = (
        "/home/marco.sertoli/data/regr_trends/"
        + datetime.datetime.now().strftime("%y_%m_%d-%H_%M")
        + "/"
    )
    if save_data:
        os.mkdir(path)

    zeff = 2.0
    wp_err = 0.2
    fabs_up = DataArray(np.linspace(0.6, 1.0), coords=[("dt", np.linspace(0.01, 0.08))])
    fabs_lo = DataArray(np.linspace(0.4, 0.8), coords=[("dt", np.linspace(0.01, 0.08))])
    fabs = (fabs_up + fabs_lo) / 2.0
    fabs_err = (fabs_up - fabs_lo) / 2.0
    hh = (8000, 9677)
    dh = (9685, 9787)
    dd = (9802, 10050)

    database = Database(reload=True)
    database.binned = general_filters(database.binned)
    database = calc_additional_quantities(database)

    pulses = DataArray(database.pulses, coords=[("pulse", database.pulses)])

    # Selection criteria for reasonable Ti
    _all = {
        "btvac_efit:value": (1.0, 2),
        "ipla_efit:value": (0.3e6, 1.0e6),
        "p_hnbi1:value": (1e3, np.nan),
        "p_rfx:value": (1.0e3, np.nan),
        "ne_smmh1:value": (2.50e19, 9.0e19),
        "vloop_l016:value": (0, np.nan),
        "te_xrcs:value": (1.0e2, 4.0e3),
        "ti_xrcs:value": (1.0e2, 10.0e3),
        "ti_xrcs:error": (np.nan, 0.3),
        "ti_xrcs:t": (0.0, np.nan),
    }
    _low_bt = deepcopy(_all)
    _low_bt["btvac_efit:value"] = (1.42, 1.47)
    _high_bt = deepcopy(_all)
    _high_bt["btvac_efit:value"] = (1.69, 1.73)
    cond = {"all": _all}  # , "Bt 1.4 T": _low_bt, "Bt 1.7 T": _high_bt}
    filtered = apply_selection(database.binned, cond)

    cond_str = {"all": "RFX & HNBI"}

    results = {}
    for k in filtered.keys():
        results[k] = {}

        binned = filtered[k]["binned"]
        _max_val, _ = calc_max_val(binned, keys="ti_xrcs")
        results[k]["max_ti"] = _max_val["ti_xrcs"]

        ifin = np.isfinite(results[k]["max_ti"].value)
        tmax = _max_val["ti_xrcs"].time
        results[k]["t_max_ti"] = tmax

        results[k]["binned"] = binned

        quantities = [
            "hnbi1_on",
            "rfx_on",
            "ipla_efit",
            "btvac_efit",
            "ti_xrcs",
            "te_xrcs",
            "ne_smmh1",
            "ne_nirh1",
            "vloop_ipla",
            "p_hnbi1",
            "p_rfx",
            "wp_efit",
            "rmag_efit",
            "rmin_efit",
            "q95_efit",
            "vloop_l016",
        ]
        for q in quantities:
            results[k][q] = xr.where(ifin, binned[q].sel(t=tmax), np.nan)

        empty = xr.full_like(results[k][q], np.nan)
        results[k]["fabs_rfx"] = deepcopy(empty)
        results[k]["fabs_rfx"].value.values = fabs.interp(
            dt=results[k]["rfx_on"].cumul
        ).values
        results[k]["fabs_rfx"].error.values = fabs_err.interp(
            dt=results[k]["rfx_on"].cumul
        ).values

        results[k]["fabs_hnbi1"] = deepcopy(empty)
        results[k]["fabs_hnbi1"].value.values = fabs.interp(
            dt=results[k]["hnbi1_on"].cumul
        ).values
        results[k]["fabs_hnbi1"].error.values = fabs_err.interp(
            dt=results[k]["hnbi1_on"].cumul
        ).values

        results[k]["zeff"] = deepcopy(empty)
        results[k]["zeff"].value.values = xr.where(ifin, zeff, np.nan).values

        results[k]["p_oh"] = deepcopy(results[k]["vloop_ipla"])

        results[k]["p_nbi"] = deepcopy(empty)
        results[k]["p_nbi"].value.values = (
            results[k]["p_rfx"].value + results[k]["p_hnbi1"].value
        ).values

        results[k]["p_nbi_abs"] = deepcopy(empty)
        results[k]["p_nbi_abs"].value.values = (
            results[k]["p_rfx"].value * results[k]["fabs_rfx"].value
            + results[k]["p_hnbi1"].value * results[k]["fabs_hnbi1"].value
        ).values

        results[k]["p_in"] = deepcopy(empty)
        results[k]["p_in"].value.values = (
            results[k]["p_oh"] + results[k]["p_nbi"]
        ).value.values

        results[k]["p_abs"] = deepcopy(empty)
        results[k]["p_abs"].value.values = (
            results[k]["p_oh"] + results[k]["p_nbi_abs"]
        ).value.values

        results[k]["wp_efit"].error.values = wp_err * results[k]["wp_efit"].value.values

        results[k]["nu_star_e"] = deepcopy(empty)
        results[k]["nu_star_e"].value.values = ph.collisionality_electrons_sauter(
            results[k]["ne_smmh1"].value.values,
            results[k]["te_xrcs"].value.values,
            results[k]["zeff"].value.values,
            results[k]["rmin_efit"].value.values,
            results[k]["rmag_efit"].value.values,
            results[k]["q95_efit"].value.values,
        )

        results[k]["tau_tot"] = deepcopy(empty)
        results[k]["tau_tot"].value.values = (
            results[k]["wp_efit"].value / (results[k]["p_in"].value)
        ).values
        results[k]["tau_tot_dwp"] = deepcopy(empty)
        results[k]["tau_tot_dwp"].value.values = (
            results[k]["wp_efit"].value
            / (results[k]["p_in"].value - results[k]["wp_efit"].gradient)
        ).values

        results[k]["tau_tot_abs"] = deepcopy(empty)
        results[k]["tau_tot_abs"].value.values = (
            results[k]["wp_efit"].value / (results[k]["p_abs"].value)
        ).values
        results[k]["tau_tot_abs_dwp"] = deepcopy(empty)
        results[k]["tau_tot_abs_dwp"].value.values = (
            results[k]["wp_efit"].value
            / (results[k]["p_abs"].value - results[k]["wp_efit"].gradient)
        ).values

        results[k]["ti_ov_te_xrcs"] = deepcopy(empty)
        results[k]["ti_ov_te_xrcs"].value.values = (
            results[k]["ti_xrcs"].value / results[k]["te_xrcs"].value
        ).values
        results[k]["ti_ov_te_xrcs"].error.values = (
            results[k]["ti_ov_te_xrcs"].value
            * np.sqrt(
                (results[k]["ti_xrcs"].error / results[k]["ti_xrcs"].value) ** 2
                + (results[k]["te_xrcs"].error / results[k]["te_xrcs"].value) ** 2
            )
        ).values

        dt_hnbi1 = (0.0, 0.2)
        dt_rfx = (0.0, 0.2)
        all_ = (
            (results[k]["rfx_on"].cumul > dt_rfx[0])
            * (results[k]["rfx_on"].cumul < dt_rfx[1])
            * (results[k]["hnbi1_on"].cumul > dt_hnbi1[0])
            * (results[k]["hnbi1_on"].cumul < dt_hnbi1[1])
        )
        dd_ = all_ * (pulses >= dd[0]) * (pulses <= dd[1])
        dh_ = all_ * (pulses >= dh[0]) * (pulses <= dh[1])
        hh_ = all_ * (pulses >= hh[0]) * (pulses <= hh[1])
        isotopes = {"HH": hh_, "DH": dh_, "DD": dd_}

        rcParams.update({"font.size": 10})
        rcParams.update({"lines.markersize": 5})

        plt.figure()
        x = results[k]["nu_star_e"].value
        y = results[k]["ti_xrcs"].value * 1.0e-3
        yerr = results[k]["ti_xrcs"].error * 1.0e-3
        for k_isotope, ind_isotope in isotopes.items():
            ind = np.where(ind_isotope.values)[0]
            plt.errorbar(
                x[ind],
                y[ind],
                yerr[ind],
                marker="o",
                alpha=0.7,
                label=k_isotope,
                linestyle="",
            )
        plt.title(k)
        plt.xlabel(r"$\nu^*$(e) Sauter")
        plt.ylabel("Ti XRCS (keV)")
        plt.ylim(0,)
        plt.xlim(0,)
        plt.legend()
        if save_data:
            save_figure(path=path, fig_name="Ti_vs_nu_star")

        plt.figure()
        x = results[k]["nu_star_e"].value
        y = results[k]["te_xrcs"].value * 1.0e-3
        yerr = results[k]["te_xrcs"].error * 1.0e-3
        for k_isotope, ind_isotope in isotopes.items():
            ind = np.where(ind_isotope.values)[0]
            plt.errorbar(
                x[ind],
                y[ind],
                yerr[ind],
                marker="o",
                alpha=0.7,
                label=k_isotope,
                linestyle="",
            )
        plt.title(k)
        plt.xlabel(r"$\nu^*$(e) Sauter")
        plt.ylabel("Te XRCS (keV)")
        plt.ylim(0,)
        plt.xlim(0,)
        plt.legend()
        if save_data:
            save_figure(path=path, fig_name="Te_vs_nu_star")

        if save_data:
            _json = json.dumps(cond[k])
            with open(f"{path}selection_criteria.json", "w") as f:
                f.write(_json)

            to_write = {}
            _all = results["all"]
            ind = np.where(np.isfinite(_all["ti_xrcs"].value.values))[0]
            to_write["Ti XRCS (eV)"] = _all["ti_xrcs"].value.values[ind]
            to_write["Ti XRCS error (eV)"] = _all["ti_xrcs"].error.values[ind]
            to_write["Te XRCS (eV)"] = _all["te_xrcs"].value.values[ind]
            to_write["Te XRCS error (eV)"] = _all["te_xrcs"].error.values[ind]
            to_write["Ip EFIT (A)"] = _all["ipla_efit"].value.values[ind]
            to_write["Bt vac EFIT (T)"] = _all["btvac_efit"].value.values[ind]
            to_write["Ne SMMH1 (m^-3)"] = _all["ne_smmh1"].value.values[ind]
            to_write["Vloop L016  (V)"] = _all["vloop_l016"].value.values[ind]
            to_write["Rmag EFIT (m)"] = _all["rmag_efit"].value.values[ind]
            to_write["a EFIT (a)"] = _all["rmin_efit"].value.values[ind]
            to_write["P HNBI1 (W)"] = _all["p_hnbi1"].value.values[ind]
            to_write["P RFX (W)"] = _all["p_rfx"].value.values[ind]
            to_write["Wp EFIT (J)"] = _all["wp_efit"].value.values[ind]

            df = pd.DataFrame(to_write)
            df.to_csv(f"{path}dataset.csv")

    return database, results

    # ind = np.where(np.isfinite(ti_xrcs.value))[0]
    # to_write = {
    #     "pulse": t_max_ti.pulse,
    #     "Ti max (eV)": ti_xrcs.values[ind],
    #     "Error of Ti max (eV)": ti_xrcs.values[ind],
    #     "Time (s) of Ti max": ti_xrcs.time[ind],
    #     "Ip (A) at time of max Ti": ti_xrcs.values[ind],
    #     "NBI power (W) at time of max Ti": nbi_at_max,
    #     "Ip max (A)": ipla_max,
    #     "Time (s) of Ip max": ipla_max_time,
    #     "Ti (0) (keV)": ti_0,
    # }
    # df = pd.DataFrame(to_write)
    # df.to_csv()


def save_figure(fig_name="", path="", orientation="landscape", ext=".jpg"):
    _file = f"{path}{fig_name}{ext}"
    plt.savefig(
        _file, orientation=orientation, dpi=600, pil_kwargs={"quality": 95},
    )
    print(f"Saving picture to {_file}")


def simulate_xrcs_2022(
    pickle_file="XRCS_temperature_parametrization_2022.pkl", write=False
):
    print("Simulating XRCS measurement for Te(0) re-scaling")

    from indica.readers import ADASReader
    from hda.diagnostics.spectrometer import XRCSpectrometer
    import hda.profiles as profiles
    import xarray as xr

    adasreader = ADASReader()
    xrcs = XRCSpectrometer(marchuk=True, extrapolate=None)

    tmp = profiles.profile_scans()
    Te_peaked = tmp["Te"]["peaked"]
    Te_broad = tmp["Te"]["broad"]
    Ne_peaked = tmp["Ne"]["peaked"]
    Ne_broad = tmp["Ne"]["broad"]
    Nimp_peaked = tmp["Nimp"]["peaked"]
    Nimp_flat = tmp["Nimp"]["flat"]
    profs = [
        {"Te": Te_broad, "Ne": Ne_broad, "Nimp": Nimp_peaked},
        {"Te": Te_broad, "Ne": Ne_peaked, "Nimp": Nimp_peaked},
        {"Te": Te_broad, "Ne": Ne_peaked, "Nimp": Nimp_flat},
        {"Te": Te_peaked, "Ne": Ne_broad, "Nimp": Nimp_peaked},
        {"Te": Te_peaked, "Ne": Ne_peaked, "Nimp": Nimp_peaked},
        {"Te": Te_peaked, "Ne": Ne_peaked, "Nimp": Nimp_flat},
    ]

    time = np.linspace(0, 1, 50)
    te_0 = np.linspace(0.5e3, 8.0e3, 50)  # central electron temperature
    ti_0 = np.linspace(0.5e3, 8.0e3, 50)  # central ion temperature
    te_sep = 50  # separatrix electron temperature
    ti_sep = 50  # separatrix ion temperature

    # Test two different profile shapes: flat (Ohmic) and slightly peaked (NBI)
    combinations = []
    for i in range(len(combinations)):
        Te_tmp = []
        for it, t in enumerate(time):
            profs[i]["Te"].y0 = te_0[it]
            profs[i]["Te"].y1 = te_sep
            profs[i]["Te"].build_profile()
            Te_tmp.append(profs[i]["Te"].yspl)

    return
    # ...

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
