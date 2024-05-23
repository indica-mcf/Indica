from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr

from indica.equilibrium import Equilibrium
from indica.readers.read_st40 import ReadST40, bin_data_in_time
from indica.utilities import FIG_PATH
from indica.utilities import save_figure
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.utilities import set_plot_rcparams
import indica.workflows.load_modelling_plasma as load_modelling
from indica.workflows.load_modelling_plasma import initialize_diagnostic_models
from indica.workflows.load_modelling_plasma import plasma_code

plt.ion()

CMAP, COLORS = set_plot_colors()


def read_modelling_runs(
    tstart: float = 0.0,
    tend: float = 0.2,
    dt: float = 0.01,
    pulse: int = 13110009,
    code: str = "astra",
    runs: list = None,
):

    if runs is None:
        runs = [f"RUN{run}" for run in (500 + np.arange(61, 77))]

    print(f"Reading {code} data")
    code_raw_data: dict = {}
    code_binned_data: dict = {}
    # equilibrium: dict = {}
    code_reader = ReadST40(pulse, tstart, tend, dt=dt, tree=code)
    for run in runs:
        code_raw_data[run] = code_reader.get_raw_data("", code, run)
        code_binned_data[run] = bin_data_in_time(
            code_reader.raw_data, tstart, tend, dt
        )

    return code_raw_data, code_binned_data


def plot_iaea_tm_2023(data=None, savefig=False, ext="png"):

    pulses = [9539, 9783, 10009]

    data = compare_pulses(
        pulses,
        data=data,
        qpop=[
            "mhd:ampl_odd_n",
            "mag:vloop",
            "diode_detr:filter_001",
            "xrcs:te_kw",
            "xrcs:ti_w",
            "cxrs_pi:ti",
            "cxrs_pi:vtor",
        ],
        savefig=savefig,
        ext=ext,
    )

    #
    # data = compare_pulses(
    #     pulses,
    #     data=data,
    #     qpop=[
    #         # "diode_detr:filter_001",
    #         "mhd:ampl_odd_n",
    #         "mag:vloop",
    #         "efit:ipla",
    #         "efit:wp",
    #         "smmh1:ne_bar",
    #         "lines:h_alpha",
    #         "nbi:pin",
    #     ],
    #     figname="kinetics",
    #     savefig=savefig,
    #     ext=ext,
    # )
    #
    return data


def plot_stan_ppcf_time_evol(data=None, savefig=False, ext="png"):

    pulses = [9520, 9539, 9780, 10009]

    data = compare_pulses(
        pulses,
        data=data,
        qpop=[
            "efit:ipla",
            "efit:wp",
            "nbi:pin",
            "mag:vloop",
            "smmh1:ne_bar",
            "smmh1:ne",
            "nirh1:ne",
            "halpha:brightness",
            "brems:brightness",
            "mhd:ampl_odd_n",
        ],
        savefig=savefig,
        ext=ext,
        add_pulse_label_to="cxff_pi:vtor",  # "sxr_diode_1:brightness",
    )

    return data


def build_astra_profiles(
    pulse: int = 9520,
    runs: list = None,
    plot: bool = True,
    save_fig: bool = False,
    write_csv: bool = False,
):
    tstart = 0.02
    tend = 0.12
    dt = 0.01
    runs_all = {
        9520: ["RUN561", "RUN562", "RUN569", "RUN570"],
        9539: ["RUN562", "RUN564", "RUN569", "RUN570", "RUN572"],
        9780: ["RUN564", "RUN566", "RUN572", "RUN573", "RUN574"],
        10009: ["RUN560", "RUN564", "RUN565", "RUN573"],
    }
    code = "astra"
    equil = "efit"  # code
    verbose = False
    instruments = ["xrcs", "cxff_pi", "smmh1", "nirh1", "efit"]

    pulse_code = int(13.1e6 + pulse)

    if pulse in runs_all.keys():
        runs = runs_all[pulse]
    else:
        raise ValueError("Pulse not in runs dictionary")

    code_raw_data, code_binned_data, tstart, tend = load_modelling.read_modelling_runs(
        code,
        pulse_code,
        runs,
        tstart=tstart,
        tend=tend,
        dt=dt,
    )
    runs = list(code_raw_data.keys())

    if pulse is not None:
        print("Reading ST40 data")
        st40 = ReadST40(pulse, tstart, tend, dt=dt, tree="st40")
        st40(instruments=instruments, map_diagnostics=False)

    if equil != code:
        equilibrium = {run: Equilibrium(st40.raw_data[equil]) for run in runs}
    else:
        equilibrium = {run: Equilibrium(code_raw_data[run]) for run in runs}

    plasma = plasma_code(
        pulse_code,
        tstart,
        tend,
        dt,
        code_raw_data,
        verbose=verbose,
        equilibrium=equilibrium,
    )

    print("Initializing diagnostic models")
    models_to_run = initialize_diagnostic_models(st40.binned_data)

    models: dict = {}
    bckc: dict = {}
    print("Running diagnostic models")
    for run in runs:
        print(run)
        bckc[run] = {}
        models[run] = {}
        _plasma = plasma[run]
        for instrument in models_to_run.keys():
            if hasattr(models_to_run[instrument], "los_transform"):
                models_to_run[instrument].los_transform.set_equilibrium(
                    equilibrium[run], force=True
                )
            if hasattr(models_to_run[instrument], "transect_transform"):
                models_to_run[instrument].transect_transform.set_equilibrium(
                    equilibrium[run], force=True
                )

            models_to_run[instrument].set_plasma(_plasma)

            _bckc = models_to_run[instrument](moment_analysis=True)
            bckc[run][instrument] = deepcopy(_bckc)
            # models[run][instrument] = deepcopy(models_to_run[instrument])

    if plot:
        plot_stan_ppcf_profiles(
            st40, bckc, plasma, save_fig=save_fig, write_csv=write_csv
        )

    return st40, bckc, models_to_run, plasma


def plot_stan_ppcf_profiles(
    st40=None,
    bckc=None,
    plasma=None,
    tplot: float = None,
    ylim: tuple = None,
    save_fig: bool = False,
    write_csv: bool = False,
):
    if st40 is None or bckc is None or plasma is None:
        st40, bckc, models_to_run, plasma = build_astra_profiles()

    CMAP, COLORS = set_plot_colors()
    set_plot_rcparams("profiles")
    cxff_pi_chans_all = {
        9520: slice(2, 3),
        9539: slice(2, 3),
        9780: slice(2, 3),
        10009: slice(2, 3),
    }
    R_shift_all = {
        9520: [-0.01, 0.01],
        9539: [-0.02, 0.0],
        9780: [0.0, 0.02],
        10009: [0.00, 0.02],
    }
    tplot_all = {9520: 0.09, 9539: 0.071, 9780: 0.081, 10009: 0.059}
    te_key = {9520: "mean", 9539: "mean", 9780: "n3w", 10009: "mean"}
    if tplot is None and st40.pulse in tplot_all.keys():
        tplot = tplot_all[st40.pulse]
        R_shift_scan = R_shift_all[st40.pulse]
        cxff_pi_chans = cxff_pi_chans_all[st40.pulse]
    if tplot is None:
        raise ValueError("Enter valid tplot")

    __Ne = []
    __Te = []
    __Ti = []
    __Ti_w_pos = []
    __Ti_w_pos_err_in = []
    __Ti_w_pos_err_out = []
    runs = list(bckc)
    for run in runs:
        t = plasma[run].t.sel(t=tplot, method="nearest")
        __Ne.append(plasma[run].electron_density.sel(t=t))
        __Te.append(plasma[run].electron_temperature.sel(t=t))
        __Ti.append(
            plasma[run].ion_temperature.sel(element=plasma[run].main_ion).sel(t=t)
        )
        __Ti_w_pos.append(bckc[run]["xrcs"]["ti_w"].pos.sel(t=t))
        __Ti_w_pos_err_in.append(bckc[run]["xrcs"]["ti_w"].pos_err_in.sel(t=t))
        __Ti_w_pos_err_out.append(bckc[run]["xrcs"]["ti_w"].pos_err_out.sel(t=t))

    _Ne = xr.concat(__Ne, "run").assign_coords(run=runs)
    _Te = xr.concat(__Te, "run").assign_coords(run=runs)
    _Ti = xr.concat(__Ti, "run").assign_coords(run=runs)
    _Ti_w_pos = xr.concat(__Ti_w_pos, "run").assign_coords(run=runs)
    _Ti_w_pos_err_in = xr.concat(__Ti_w_pos_err_in, "run").assign_coords(run=runs)
    _Ti_w_pos_err_out = xr.concat(__Ti_w_pos_err_out, "run").assign_coords(run=runs)

    Ne = _Ne.mean("run")
    Te = _Te.mean("run")
    Ti = _Ti.mean("run")
    Ti_w = st40.raw_data["xrcs"]["ti_w"].sel(t=tplot, method="nearest")
    Ti_w_err = st40.raw_data["xrcs"]["ti_w"].error.sel(t=tplot, method="nearest")
    Te_kw = st40.raw_data["xrcs"]["te_kw"].sel(t=tplot, method="nearest")
    Te_kw_err = st40.raw_data["xrcs"]["te_kw"].error.sel(t=tplot, method="nearest")
    Te_n3w = st40.raw_data["xrcs"]["te_n3w"].sel(t=tplot, method="nearest")
    Te_n3w_err = st40.raw_data["xrcs"]["te_n3w"].error.sel(t=tplot, method="nearest")
    Ti_cxff_pi = (
        st40.raw_data["cxff_pi"]["ti"]
        .sel(t=tplot, method="nearest")
        .sel(channel=cxff_pi_chans)
    )
    Ti_cxff_pi_err = (
        st40.raw_data["cxff_pi"]["ti"]
        .error.sel(t=tplot, method="nearest")
        .sel(channel=cxff_pi_chans)
    )

    _Ti_cxff_pi_pos_all = []
    for R_shift in R_shift_scan:
        R = Ti_cxff_pi.R + R_shift
        z = Ti_cxff_pi.z * 0
        pos, _, _ = plasma[run].equilibrium.flux_coords(R, z, Ti_cxff_pi.t)
        _Ti_cxff_pi_pos_all.append(pos)

    Ti_cxff_pi_pos_all = xr.concat(_Ti_cxff_pi_pos_all, "R_shift").assign_coords(
        R_shift=R_shift_scan
    )

    Te_mean = np.mean([Te_kw, Te_n3w])
    err_kw = (Te_kw_err / Te_kw) ** 2
    err_n3w = (Te_n3w_err / Te_n3w) ** 2
    Te_mean_err = err_kw * 0
    if err_kw < 0.2:
        Te_mean_err += err_kw
    if err_n3w < 0.2:
        Te_mean_err += err_n3w
    Te_mean_err = np.sqrt(Te_mean_err) * Te_mean
    if te_key[st40.pulse] == "mean":
        Te_xrcs = Te_mean
        Te_xrcs_err = Te_mean_err
    elif te_key[st40.pulse] == "kw":
        Te_xrcs = Te_kw
        Te_xrcs_err = Te_kw_err
    elif te_key[st40.pulse] == "n3w":
        Te_xrcs = Te_n3w
        Te_xrcs_err = Te_n3w_err

    Ti_w_pos = _Ti_w_pos.mean("run")
    Ti_w_pos_err_in = _Ti_w_pos_err_in.max("run")
    Ti_w_pos_err_out = _Ti_w_pos_err_out.max("run")

    Ne_err = _Ne.std("run")
    Te_err = _Te.std("run")
    Ti_err = _Ti.std("run")

    plt.figure()
    plt.plot(Ti.rho_poloidal, Ti, color=COLORS["ion"], label="Ions")
    plt.fill_between(Ti.rho_poloidal, Ti - Ti_err, Ti + Ti_err, color=COLORS["ion"])
    plt.plot(Te.rho_poloidal, Te, color=COLORS["electron"], label="Electrons")
    plt.fill_between(
        Te.rho_poloidal, Te - Te_err, Te + Te_err, color=COLORS["electron"]
    )

    plt.hlines(
        Te_xrcs,
        Ti_w_pos - Ti_w_pos_err_in,
        Ti_w_pos + Ti_w_pos_err_out,
        color="white",
        linewidth=plt.rcParams["lines.linewidth"] * 1.5,
    )
    plt.hlines(
        Te_xrcs,
        Ti_w_pos - Ti_w_pos_err_in,
        Ti_w_pos + Ti_w_pos_err_out,
        color=COLORS["electron"],
    )
    plt.vlines(
        Ti_w_pos,
        Te_xrcs - Te_xrcs_err,
        Te_xrcs + Te_xrcs_err,
        color="white",
        linewidth=plt.rcParams["lines.linewidth"] * 1.5,
    )
    plt.vlines(
        Ti_w_pos, Te_xrcs - Te_xrcs_err, Te_xrcs + Te_xrcs_err, color=COLORS["electron"]
    )
    plt.scatter(
        Ti_w_pos,
        Te_xrcs,
        marker="o",
        color=COLORS["electron"],
        facecolor="white",
        label="XRCS",
        zorder=3,
    )

    plt.hlines(
        Ti_w,
        Ti_w_pos - Ti_w_pos_err_in,
        Ti_w_pos + Ti_w_pos_err_out,
        color="white",
        linewidth=plt.rcParams["lines.linewidth"] * 1.5,
    )
    plt.hlines(
        Ti_w,
        Ti_w_pos - Ti_w_pos_err_in,
        Ti_w_pos + Ti_w_pos_err_out,
        color=COLORS["ion"],
    )
    plt.vlines(
        Ti_w_pos,
        Ti_w - Ti_w_err,
        Ti_w + Ti_w_err,
        color="white",
        linewidth=plt.rcParams["lines.linewidth"] * 1.5,
    )
    plt.vlines(Ti_w_pos, Ti_w - Ti_w_err, Ti_w + Ti_w_err, color=COLORS["ion"])
    plt.scatter(
        Ti_w_pos,
        Ti_w,
        marker="o",
        color=COLORS["ion"],
        facecolor="white",
        zorder=3,
    )

    Ti_cxff_pi_pos = Ti_cxff_pi_pos_all.mean("R_shift")
    Ti_cxff_pi_pos_err = Ti_cxff_pi_pos_all.max("R_shift") - Ti_cxff_pi_pos_all.min(
        "R_shift"
    )
    plt.hlines(
        Ti_cxff_pi,
        Ti_cxff_pi_pos - Ti_cxff_pi_pos_err,
        Ti_cxff_pi_pos + Ti_cxff_pi_pos_err,
        color="white",
        linewidth=plt.rcParams["lines.linewidth"] * 1.5,
    )
    plt.hlines(
        Ti_cxff_pi,
        Ti_cxff_pi_pos - Ti_cxff_pi_pos_err,
        Ti_cxff_pi_pos + Ti_cxff_pi_pos_err,
        color=COLORS["ion"],
    )
    plt.vlines(
        Ti_cxff_pi_pos,
        Ti_cxff_pi - Ti_cxff_pi_err,
        Ti_cxff_pi + Ti_cxff_pi_err,
        color="white",
        linewidth=plt.rcParams["lines.linewidth"] * 1.5,
    )
    plt.vlines(
        Ti_cxff_pi_pos,
        Ti_cxff_pi - Ti_cxff_pi_err,
        Ti_cxff_pi + Ti_cxff_pi_err,
        color=COLORS["ion"],
    )
    plt.scatter(
        Ti_cxff_pi_pos,
        Ti_cxff_pi,
        marker="s",
        color=COLORS["ion"],
        facecolor="white",
        label="CXRS",
        zorder=3,
    )
    set_axis_sci()
    plt.xlabel(r"$\rho_{pol}$")
    plt.ylabel("[keV]")
    plt.legend()
    plt.title(f"{st40.pulse} @ t={tplot:1.2f}s")
    if ylim is not None:
        plt.ylim(ylim)

    if save_fig:
        save_figure(FIG_PATH, f"{st40.pulse}_Stan_PPCF_figure", save_fig=save_fig)

    plt.figure()
    plt.vlines(tplot, 0, 1.0e4)
    st40.raw_data["xrcs"]["ti_w"].plot(marker="o", label="Ti_w")
    st40.raw_data["xrcs"]["te_kw"].plot(marker="o", label="Te_kw")
    st40.raw_data["xrcs"]["te_n3w"].plot(marker="o", label="Te_n3w")
    for chan in st40.raw_data["cxff_pi"]["ti"].sel(channel=cxff_pi_chans).channel:
        print(
            chan.values,
            Ti_cxff_pi.R.sel(channel=chan).values + np.mean(R_shift_scan),
            plasma[run].equilibrium.rmag.sel(t=tplot, method="nearest").values,
        )
        st40.raw_data["cxff_pi"]["ti"].sel(channel=chan).plot(
            marker="s", label="Ti_cxrs"
        )
    plt.legend()

    if write_csv:
        to_write = {
            "Rho-poloidal": Te.rho_poloidal.values,
            "Ne value (m**-3)": Ne.values,
            "Ne error (m**-3)": Ne_err.values,
            "Te value (eV)": Te.values,
            "Te error (eV)": Te_err.values,
            "Ti value (eV)": Ti.values,
            "Ti error (eV)": Ti_err.values,
        }
        df = pd.DataFrame(to_write)
        df.to_csv(f"{FIG_PATH}{st40.pulse}_{t.values:1.3f}s_HDA_profiles.csv")

    return

    # if plot or save_fig:
    #     load_modelling.plot_plasmas(
    #         plasma,
    #         tplot,
    #         code=code,
    #         save_fig=save_fig,
    #         fig_style=fig_style,
    #         alpha=alpha,
    #     )
    #
    #     load_modelling.plot_data_bckc_comparison(
    #         st40,
    #         bckc,
    #         plasma,
    #         tplot,
    #         code=code,
    #         save_fig=save_fig,
    #         fig_style=fig_style,
    #         alpha=alpha,
    #     )

    # return st40, bckc, plasma


def smmh1_evolution(data=None, savefig=False, ext="png"):

    # pulses = [9520, 9780, 9850, 10009]
    pulses = [10619, 10620, 10605, 10607]

    data = compare_pulses(
        pulses,
        data=data,
        qpop=[
            "nbi:pin",
            "mag:vloop",
            "smmh1:ne_bar",
            "halpha:brightness",
            "sxr_diode_1:brightness",
            "brems:brightness",
            "cxff_pi:ti",
            "cxff_pi:vtor",
            "xrcs:te_kw",
            "xrcs:ti_w",
            "mhd:ampl_odd_n",
        ],
        savefig=savefig,
        ext=ext,
        add_pulse_label_to="efit:ipla",
    )

    return data


def compare_pulses(
    pulses: list = [9538, 9780, 9783, 9831, 10014],
    tstart: float = 0.001,
    tend: float = 0.15,
    alpha: float = 0.9,
    xlabel: str = "Time (s)",
    figname="",
    savefig: bool = False,
    qpop: list = [""],
    data: list = None,
    R_cxrs: float = 0.4649,
    ext: str = "png",
    add_pulse_label_to="nbi:pin",
):
    """
    Compare time traces of different pulses
    for APS:
    compare_pulses(qpop=["lines:h_alpha", "mhd:ampl_odd_n"])
    """

    qdict = available_quantities()
    for q in qpop:
        if q in qdict.keys():
            qdict.pop(q)

    iax = -1
    set_plot_rcparams("time_evolution")

    hh = range(8000, 9677 + 1)
    dh = range(9685, 9787 + 1)
    dd = range(9802, 10050 + 1)
    pulse_labels = []
    for p in pulses:
        if p in hh:
            pulse_labels.append("HH")
        elif p in dh:
            pulse_labels.append("DH")
        elif p in dd:
            pulse_labels.append("DD")
        else:
            pulse_labels.append("")

    instruments = [q.split(":")[0] for q in qdict.keys()]
    if data is None:
        data = []
        for pulse in pulses:
            print(pulse)
            st40_data = ReadST40(pulse, tstart, tend)
            st40_data(instruments=instruments, raw_only=True)
            raw_data = st40_data.raw_data

            if "cxff_pi" in raw_data.keys():
                chan = np.argmin(np.abs(raw_data["cxff_pi"]["ti"].R - R_cxrs).values)
                _data = raw_data["cxff_pi"]["ti"].sel(channel=chan)
                _data = xr.where(_data > 0, _data, np.nan)
                _error = raw_data["cxff_pi"]["ti"].error.sel(channel=chan)
                _error = xr.where(_error > 0, _error, np.nan)
                raw_data["cxff_pi"]["ti"] = _data
                raw_data["cxff_pi"]["ti"].attrs["error"] = _error

                _data = raw_data["cxff_pi"]["vtor"].sel(channel=chan)
                _data = xr.where(_data > 0, _data, np.nan)
                _error = raw_data["cxff_pi"]["vtor"].error.sel(channel=chan)
                _error = xr.where(_error > 0, _error, np.nan)
                raw_data["cxff_pi"]["vtor"] = _data
                raw_data["cxff_pi"]["vtor"].attrs["error"] = _error

            if "hnbi1" in raw_data.keys():
                raw_data["nbi"] = {
                    "pin": raw_data["hnbi1"]["pin"] + raw_data["rfx"]["pin"]
                }
            if "efit" in raw_data.keys():
                smmh1_l = 2 * (raw_data["efit"]["rmjo"] - raw_data["efit"]["rmji"]).sel(
                    rho_poloidal=1
                )
                raw_data["smmh1"]["ne_bar"] = raw_data["smmh1"]["ne"] / smmh1_l.interp(
                    t=raw_data["smmh1"]["ne"].t
                )

            data.append(raw_data)

    fig, axs = plt.subplots(len(qdict.keys()), figsize=(6, 8))

    for key in qdict.keys():
        print(key)
        iax += 1
        ax = axs[iax]
        plot_quantity(
            st40_data,
            ax,
            data,
            key,
            pulses,
            qdict,
            add_pulse_label_to=add_pulse_label_to,
            alpha=alpha,
        )
        if iax != (len(axs) - 1):
            ax.xaxis.set_ticklabels([])
        else:
            ax.set_xlabel(xlabel)

    if savefig:
        name = ""
        for pulse in pulses:
            name += f"_{pulse}"
        if len(figname) > 1:
            name += f"_{figname}"
        save_figure(
            path_name=FIG_PATH, fig_name=f"time_evolution_comparison{name}", ext=ext
        )

    return data


def plot_quantity(
    st40_data: ReadST40,
    ax: plt.axis,
    data: list,
    qkey: str,
    pulses: list,
    qdict: dict,
    linestyle="solid",
    add_pulse_label_to: str = "",
    alpha: float = 0.9,
):

    ncols = 4
    if len(pulses) > ncols:
        ncols = len(pulses)
    cols = CMAP(np.linspace(0.1, 0.75, ncols, dtype=float))

    if "label" not in qdict[qkey]:
        qdict[qkey]["label"] = ""
    if "marker" not in qdict[qkey].keys():
        qdict[qkey]["marker"] = ""

    diag, quant = qkey.split(":")
    label = ""
    add_pulse_label = False
    if qkey == add_pulse_label_to:
        add_pulse_label = True
    for i in range(len(data)):
        skip = False
        if diag not in data[i].keys():
            skip = True
            continue
        if quant not in data[i][diag].keys():
            skip = True
            continue

        if add_pulse_label:
            label = str(pulses[i])
        val = data[i][diag][quant] * qdict[qkey]["const"]

        if "error" in data[i][diag][quant].attrs.keys():
            err = data[i][diag][quant].error * qdict[qkey]["const"]
            ind = np.where(np.isfinite(val.values) * np.isfinite(err.values))[0]
            ax.errorbar(
                val.t[ind],
                val.values[ind],
                err.values[ind],
                alpha=alpha,
                color=cols[i],
                marker=qdict[qkey]["marker"],
                label=label,
                linestyle=linestyle,
            )
        else:
            ind = np.where(np.isfinite(val.values))[0]
            ax.plot(
                val.t[ind],
                val.values[ind],
                alpha=alpha,
                color=cols[i],
                label=label,
                marker=qdict[qkey]["marker"],
                linestyle=linestyle,
            )

    if skip:
        return

    ax.set_ylabel(qdict[qkey]["ylabel"])

    # add pulses label
    if add_pulse_label:
        loc = "upper right"
        plt.gca()
        ax.legend(frameon=True, handlelength=None, loc=loc)

    # add quantity label
    if len(qdict[qkey]["label"]) > 0:
        ax_label = ax.twinx()
        ax_label.plot(
            [np.nan],
            [np.nan],
            label=qdict[qkey]["label"],
        )
        ax_label.get_yaxis().set_visible(False)
        ax_label.legend(frameon=False, handlelength=0, loc="upper left")

    ax.set_xlim(st40_data.tstart, st40_data.tend)
    if "ylim" in qdict[qkey].keys():
        ylim = qdict[qkey]["ylim"]
    ax.set_ylim(ylim)


def available_quantities():

    qdict = {
        "efit:ipla": {
            "const": 1.0e-6,
            "label": "$I_P$ $EFIT$",
            "ylabel": "$(MA)$",
            "ylim": (0, None),
        },
        "efit:wp": {
            "const": 1.0e-3,
            "label": "$W_P$ $EFIT$",
            "ylabel": "$(kJ)$",
            "ylim": (0, None),
        },
        "nbi:pin": {
            "const": 1.0e-6,
            "label": "$P_{NBI}$",
            "ylabel": "$(MW)$",
            "ylim": (0, None),
        },
        "mag:vloop": {
            "const": 1.0,
            "ylabel": "$V_{loop}$ $(V)$",
            "ylim": (0, None),
        },
        "smmh1:ne_bar": {
            "const": 1.0e-19,
            "label": r"$\overline{N}_e$ $SMM$",
            "ylabel": "($10^{19}$ $m^{-3}$)",
            "ylim": (0, 7),
        },
        "smmh1:ne": {
            "const": 1.0e-19,
            "label": "$N_e$-int SMM",
            "ylabel": "($10^{19}$ $m^{-2}$)",
            "ylim": (0, 7),
        },
        "nirh1:ne": {
            "const": 1.0e-19,
            "label": "$N_e$-int NIR",
            "ylabel": "($10^{19}$ $m^{-2}$)",
            "ylim": (0, 15),
        },
        "halpha:brightness": {
            "const": 1.0,
            "label": r"$H_\alpha$ $Filter$",
            "ylabel": r"($a.u.)$",
            "ylim": (0, 0.5),
        },
        "brems:brightness": {
            "const": 1.0,
            "label": r"$Bremsstrahlung $Filter$",
            "ylabel": r"($a.u.)$",
            "ylim": (0, 0.5),
        },
        "sxr_diode_1:brightness": {
            "const": 1.0e-3,
            "label": "$P_{SXR}$",
            "ylabel": "$(a.u.)$",
            "ylim": (0, None),
        },
        "xrcs:te_kw": {
            "const": 1.0e-3,
            "label": "$T_e$ $XRCS$",
            "ylabel": "$(keV)$",
            "ylim": (0, None),
            "error": True,
            "marker": "o",
        },
        "xrcs:ti_w": {
            "const": 1.0e-3,
            "label": "$T_i$ $XRCS$",
            "ylabel": "$(keV)$",
            "ylim": (0, 10),
            "error": True,
            "marker": "o",
        },
        "cxff_pi:ti": {
            "const": 1.0e-3,
            "label": "$T_i$ $CXRS$",
            "ylabel": "$(keV)$",
            "ylim": (0, 10),
            "error": True,
            "marker": "x",
        },
        "cxff_pi:vtor": {
            "const": 1.0e-3,
            "label": "$V_{tor}$ $CXRS$",
            "ylabel": "$(km/s)$",
            "ylim": (0, 300),
            "error": True,
            "marker": "x",
        },
        "mhd:ampl_odd_n": {
            "const": 1.0,
            "label": "$Odd$ $n$ $MHD$",
            "ylabel": "$(a.u.)$",
            "ylim": (0, None),
        },
    }

    return qdict
