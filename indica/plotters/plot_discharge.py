from hda.read_st40 import ST40data
from matplotlib import cm
from matplotlib import rcParams
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.time import bin_in_time_dt
from indica.equilibrium import Equilibrium

plt.ion()


def set_sizes(fontsize=9, legendsize=7, markersize=3):
    rcParams.update({"font.size": fontsize})
    rcParams.update({"font.size": fontsize})
    rcParams.update({"legend.fontsize": legendsize})
    rcParams.update({"lines.markersize": markersize})


def available_quantities():

    qdict = {
        "efit:ipla": {
            "const": 1.0e-6,
            "label": "EFIT",
            "ylabel": "$I_P$ (MA)",
            "ylim": (0, None),
        },
        "efit:wp": {
            "const": 1.0e-3,
            "label": "EFIT",
            "ylabel": "$W_P$ (kJ)",
            "ylim": (0, None),
        },
        "nbi:pin": {
            "const": 1.0e-6,
            "label": "",
            "ylabel": "$P_{NBI}$ (MW)",
            "ylim": (0, None),
        },
        "mag:vloop": {
            "const": 1.0,
            "ylabel": "$V_{loop}$ (V)",
            "ylim": (0, None),
        },
        "smmh1:ne": {
            "const": 1.0e-19,
            "label": "SMM",
            "ylabel": "$N_e$ ($10^{19}$ $m^{-3}$)",
            "ylim": (0, None),
        },
        "xrcs:te_avrg": {
            "const": 1.0e-3,
            "label": "XRCS(Ar)",
            "ylabel": "$T_e$ (keV)",
            "ylim": (0, None),
            "error": True,
            "marker": "o",
        },
        "xrcs:ti_w": {
            "const": 1.0e-3,
            "label": "XRCS(Ar)",
            "ylabel": "$T_i$ (keV)",
            "ylim": (0, None),
            "error": True,
            "marker": "o",
        },
        "diode_detr:filter_001": {
            "const": 1.0e-3,
            "label": "",
            "ylabel": "$SXR$ (a.u.)",
            "ylim": (0, None),
        },
        "lines:h_alpha": {
            "const": 1.0,
            "label": "H-alpha Filter",
            "ylabel": r"$H_\alpha$ (a.u.)",
            "ylim": (0, None),
        },
        "mhd:ampl_odd_n": {
            "const": 1.0,
            "label": "MHD",
            "ylabel": "Odd n",
            "ylim": (0, None),
        },
    }

    return qdict


def add_cxrs(st40_data: ST40data, raw_data: dict):

    raw_data["cxrs"] = {}

    print("Reading full-fit CXSFIT analysis")
    if st40_data.pulse == 10014:
        rev = 7
        t_slice = slice(0.04, st40_data.tend)
    elif st40_data.pulse == 9780:
        rev = 5
        t_slice = slice(st40_data.tstart, st40_data.tend)
    else:
        print("Revision for CXRS not defined")
        return

    ti, dims = st40_data.reader._get_data(
        "spectrom", "princeton.cxsfit_out", ":ti", rev
    )
    ti_error, dims = st40_data.reader._get_data(
        "spectrom", "princeton.cxsfit_out", ":ti_err", rev
    )

    R_slice = slice(0.466, 0.467)
    ti_cx = (
        DataArray(ti, coords=[("t", dims[1]), ("R", dims[0])])
        .sel(t=t_slice)
        .sortby("R")
    )
    ti_err_cx = (
        DataArray(ti_error, coords=[("t", dims[1]), ("R", dims[0])])
        .sel(t=t_slice)
        .sortby("R")
    )
    ti_err_cx = xr.where(ti_cx > 0, ti_err_cx, np.nan)
    ti_cx = xr.where(ti_cx > 0, ti_cx, np.nan)

    ti_cx_max = ti_cx.sel(R=R_slice)  # R=0.4663918)
    ti_err_cx_max = ti_err_cx.sel(R=R_slice)  # R=0.4663918)
    ti_err_cx_max = xr.where(ti_cx_max > 0, ti_err_cx_max, np.nan)
    ti_cx_max = xr.where(ti_cx_max > 0, ti_cx_max, np.nan)

    ti_cx.attrs = {"error": ti_err_cx}
    ti_cx_max.attrs = {"error": ti_err_cx_max}

    raw_data["cxrs"]["ti_full"] = ti_cx
    raw_data["cxrs"]["ti_full_max"] = ti_cx_max

    print("Reading background-subtraction CXSFIT analysis")
    if st40_data.pulse == 10014:
        rev = 9
        t_slice = slice(0.04, st40_data.tend)
    elif st40_data.pulse == 9780:
        # raise ValueError("What's the best run for 9780 bgnd-subtraction?")
        rev = 5
        t_slice = slice(st40_data.tstart, st40_data.tend)
    else:
        print("Revision for CXRS not defined")
        return

    ti, dims = st40_data.reader._get_data(
        "spectrom", "princeton.cxsfit_out", ":ti", rev
    )
    ti_error, dims = st40_data.reader._get_data(
        "spectrom", "princeton.cxsfit_out", ":ti_err", rev
    )

    R_slice = slice(0.466, 0.467)
    ti_cx = (
        DataArray(ti, coords=[("t", dims[1]), ("R", dims[0])])
        .sel(t=t_slice)
        .sortby("R")
    )
    ti_err_cx = (
        DataArray(ti_error, coords=[("t", dims[1]), ("R", dims[0])])
        .sel(t=t_slice)
        .sortby("R")
    )
    ti_err_cx = xr.where(ti_cx > 0, ti_err_cx, np.nan)
    ti_cx = xr.where(ti_cx > 0, ti_cx, np.nan)

    ti_cx_max = ti_cx.sel(R=R_slice)  # R=0.4663918)
    ti_err_cx_max = ti_err_cx.sel(R=R_slice)  # R=0.4663918)
    ti_err_cx_max = xr.where(ti_cx_max > 0, ti_err_cx_max, np.nan)
    ti_cx_max = xr.where(ti_cx_max > 0, ti_cx_max, np.nan)

    ti_cx.attrs = {"error": ti_err_cx}
    ti_cx_max.attrs = {"error": ti_err_cx_max}

    raw_data["cxrs"]["ti_bgnd"] = ti_cx
    raw_data["cxrs"]["ti_bgnd_max"] = ti_cx_max


def add_mhd(st40_data: ST40data, raw_data: dict):

    t_slice = slice(st40_data.tstart, st40_data.tend)
    rev = 0

    even, even_dims = st40_data.reader._get_data(
        "", "mhd_tor_mode", ".output.spectrogram:ampl_even", rev
    )
    odd, odd_dims = st40_data.reader._get_data(
        "", "mhd_tor_mode", ".output.spectrogram:ampl_odd", rev
    )

    even = DataArray(even, coords=[("t", even_dims[0])]).sel(t=t_slice)
    odd = DataArray(odd, coords=[("t", odd_dims[0])]).sel(t=t_slice)

    raw_data["mhd"] = {}
    raw_data["mhd"]["ampl_even_n"] = even
    raw_data["mhd"]["ampl_odd_n"] = odd


def add_btot(raw_data: dict):
    equilibrium = Equilibrium(raw_data["efit"])
    R = raw_data["efit"]["rmag"]
    z = raw_data["efit"]["zmag"]
    Btot = equilibrium.Btot(R, z)[0]
    raw_data["efit"]["btot"] = Btot


def compare_pulses_prl(data=None):

    data = compare_pulses(data=data, qpop=["mhd:ampl_odd_n", "lines:h_alpha"])

    return data


def compare_pulses_aps(data=None):

    # 9520, 9538, 9783, 9831

    data = compare_pulses(
        [9539, 9780, 10009], data=data, qpop=["mhd:ampl_odd_n", "lines:h_alpha"]
    )

    return data


def compare_pulses(  # 9783, 9781, 9831, 10013,
    pulses: list = [9538, 9780, 9783, 9831, 10014],
    tstart: float = -0.01,
    tend: float = 0.2,
    dt: float = 0.003,
    alpha: float = 0.7,
    xlabel: str = "Time (s)",
    savefig: bool = False,
    qpop: list = [""],
    data: list = None,
):
    """
    Compare time traces of different pulses
    for APS:
    compare_pulses(qpop=["lines:h_alpha", "mhd:ampl_odd_n"])
    """

    def plot_quantity(qkey: str, linestyle="solid"):
        if "label" not in qdict[qkey]:
            qdict[qkey]["label"] = ""
        if "marker" not in qdict[qkey].keys():
            qdict[qkey]["marker"] = ""

        handlelength = None
        frameon = True
        diag, quant = qkey.split(":")
        label = ""
        binned = np.array([])
        for i in range(len(data)):
            if type(qdict[qkey]["label"]) is list:
                label = qdict[qkey]["label"][i]
            val = data[i][diag][quant] * qdict[qkey]["const"]
            try:
                _binned = bin_in_time_dt(val.t.min() + dt, val.t.max() - dt, dt, val)
                binned = np.append(binned, _binned.values).flatten()
            except ValueError:
                _ = np.nan

            if "error" in data[i][diag][quant].attrs:
                err = data[i][diag][quant].error * qdict[qkey]["const"]
                ind = np.where(
                    np.isfinite(val.values)
                    * np.isfinite(err.values)
                    * ((err.values / val.values) < 0.2)
                )[0]
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
        if type(qdict[qkey]["label"]) is str and len(qdict[qkey]["label"]) > 0:
            handlelength = 0
            frameon = False
            ax.plot(
                val.t[ind],
                val.values[ind],
                label=qdict[qkey]["label"],
                alpha=alpha,
                color=cols[i],
            )
        ax.set_ylabel(qdict[qkey]["ylabel"])
        if len(qdict[qkey]["label"]) > 0:
            if type(qdict[qkey]["label"]) == str:
                loc = None
            else:
                loc = None
            ax.legend(frameon=frameon, handlelength=handlelength, loc=loc)
        ax.set_xlim(tstart, tend)
        if np.size(binned) > 0:
            ylim_bin = (np.min(binned) * 0.7, np.max(binned) * 1.3)
        if "ylim" in qdict[qkey].keys():
            ylim = qdict[qkey]["ylim"]
            if ylim[0] is None and np.isfinite(ylim_bin[0]):
                ylim = (ylim_bin[1], ylim[1])
            if ylim[1] is None and np.isfinite(ylim_bin[1]):
                ylim = (ylim[0], ylim_bin[1])
        else:
            ylim = ylim_bin
        ax.set_ylim(ylim)

    qdict = available_quantities()
    qdict["nbi:pin"]["label"] = pulses
    for q in qpop:
        if q in qdict.keys():
            qdict.pop(q)

    iax = -1
    set_sizes()

    cols = ["black", "blue", "magenta", "orange", "red"]
    if len(pulses) > len(cols):
        print("Missing colours...")
        cmap = cm.rainbow
        varr = np.linspace(0, 1, len(pulses))
        cols = cmap(varr)

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

    if data is None:
        data = []
        for pulse in pulses:
            print(pulse)
            st40_data = ST40data(pulse, tstart, tend)
            st40_data.get_all()
            st40_data.get_other_data()
            raw_data = st40_data.data
            add_cxrs(st40_data, raw_data)
            add_mhd(st40_data, raw_data)
            add_btot(raw_data)

            raw_data["nbi"] = {"pin": raw_data["hnbi1"]["pin"] + raw_data["rfx"]["pin"]}

            tmp = raw_data["xrcs"]["te_avrg"]
            err = raw_data["xrcs"]["te_avrg"].error
            raw_data["xrcs"]["te_avrg"].values = xr.where(err / tmp < 0.2, tmp, np.nan)

            data.append(raw_data)

    fig, axs = plt.subplots(len(qdict.keys()), figsize=(6, 8))

    for key in qdict.keys():
        iax += 1
        ax = axs[iax]
        plot_quantity(key)
        if iax != (len(axs) - 1):
            ax.xaxis.set_ticklabels([])
        else:
            ax.set_xlabel(xlabel)

        # if key == "efit:ipla":
        #     plot_quantity("nbi:pin")

    if savefig:
        figname = ""
        for pulse in pulses:
            figname += f"_{pulse}"
        save_figure(fig_name=f"time_evolution_comparison{figname}")

    return data


def data_time_evol(
    pulse,
    tstart=-0.01,
    tend=0.2,
    savefig=False,
    name="",
    title="",
    plot_100M=False,
):
    dt = 0.01
    st40_data = ST40data(pulse, tstart, tend)
    st40_data.get_all()
    st40_data.get_other_data()
    raw_data = st40_data.data
    add_cxrs(st40_data, raw_data)
    add_btot(raw_data)

    figname = get_figname(pulse=pulse, name=name)
    _title = f"Pulse {pulse}"
    if len(title) > 1:
        _title += f" {title}"

    fig, axs = plt.subplots(3, 2)
    iax0, iax1 = 0, 0
    ax = axs[iax0, iax1]
    const = 1.0e-6
    tmp = raw_data["efit"]["ipla"] * const
    ax.plot(tmp.t, tmp.values, label="I$_P$", alpha=0.9)
    binned = bin_in_time_dt(tmp.t.min() + dt / 2, tmp.t.max() - dt / 2, dt, tmp)
    ax.set_ylim(bottom=0, top=np.ceil(binned.max()))
    ax.set_ylabel("(MA)")
    ax.xaxis.set_ticklabels([])
    ax.legend()
    ax.set_xlim(tstart, tend)

    iax0, iax1 = 1, 0
    ax = axs[iax0, iax1]
    const = 1.0
    tmp = raw_data["efit"]["btot"] * const
    ax.plot(tmp.t, tmp.values, label="B$_{tot}(0)$", alpha=0.9)
    binned = bin_in_time_dt(tmp.t.min() + dt / 2, tmp.t.max() - dt / 2, dt, tmp)
    ax.set_ylim(
        bottom=np.floor(binned.min().values),
        top=np.ceil(binned.max()),
    )
    ax.set_ylabel("(T)")
    ax.xaxis.set_ticklabels([])
    ax.legend()
    ax.set_xlim(tstart, tend)

    iax0, iax1 = 2, 0
    ax = axs[iax0, iax1]
    const = 1.0e-3
    tmp = raw_data["efit"]["wp"] * const
    ax.plot(tmp.t, tmp.values, label="W$_P$", alpha=0.9)
    binned = bin_in_time_dt(tmp.t.min() + dt / 2, tmp.t.max() - dt / 2, dt, tmp)
    ax.set_ylim(bottom=0, top=np.ceil(binned.max()))
    ax.set_ylim(bottom=0, top=50)
    ax.set_ylabel("(kJ)")
    ax.legend()
    ax.set_xlim(tstart, tend)
    ax.set_xlabel("Time (s)")

    iax0, iax1 = 0, 1
    ax = axs[iax0, iax1]
    const = 1.0e-6
    if "hnbi1" in raw_data:
        tmp = raw_data["hnbi1"]["pin"] * const
        ax.plot(tmp.t, tmp.values, label="P$_{HNBI1}$", alpha=0.8)
    if "rfx" in raw_data:
        tmp = raw_data["rfx"]["pin"] * const
        ax.plot(tmp.t, tmp.values, label="P$_{RFX}$", alpha=0.8)
    const = 1.0e-19
    tmp = raw_data["smmh1"]["ne"] * const
    ax.plot(tmp.t, tmp.values, label="n$_e$", alpha=0.9)
    binned = bin_in_time_dt(tmp.t.min() + dt / 2, tmp.t.max() - dt / 2, dt, tmp)
    ax.set_ylim(bottom=0, top=np.ceil(binned.max() * 1.5))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("(10$^{19}$ m$^{-3}$) / (MW)")
    ax.legend()
    ax.set_xlim(tstart, tend)

    iax0, iax1 = 1, 1
    ax = axs[iax0, iax1]
    binned = np.array([])
    const = 1.0e3
    tmp = raw_data["lines"]["brems"] * const
    binned = np.append(
        binned, bin_in_time_dt(tmp.t.min() + dt / 2, tmp.t.max() - dt / 2, dt, tmp)
    )
    ax.plot(tmp.t, tmp.values, label="P$_{SXR}}$", alpha=0.9)
    const = 1.0e-3

    tmp = raw_data["diode_detr"]["filter_001"] * const
    binned = np.append(
        binned, bin_in_time_dt(tmp.t.min() + dt / 2, tmp.t.max() - dt / 2, dt, tmp)
    )
    ax.plot(tmp.t, tmp.values, label="P$_{Brems}}$", alpha=0.9)
    ax.set_ylim(bottom=0, top=np.ceil(binned.max() * 1.2))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("(a.u.)")
    ax.legend()
    ax.set_xlim(tstart, tend)

    iax0, iax1 = 2, 1
    ax = axs[iax0, iax1]
    const = 1.0e-3
    tmp = raw_data["xrcs"]["te_kw"] * const
    err = raw_data["xrcs"]["te_kw"].error * const
    err = xr.where(err > 0, err, np.nan)
    ax.errorbar(
        tmp.t,
        tmp.values,
        err.values,
        label="T$_e$(Ar)",
        marker="o",
        alpha=0.9,
    )

    tmp = raw_data["xrcs"]["ti_w"] * const
    err = raw_data["xrcs"]["ti_w"].error * const
    err = xr.where(err > 0, err, np.nan)
    ax.errorbar(
        tmp.t,
        tmp.values,
        err.values,
        label="T$_i$(Ar)",
        marker="o",
        alpha=0.9,
    )

    return raw_data

    tmp = raw_data["cxrs"]["ti_full_max"] * const
    err = raw_data["cxrs"]["ti_full_max"].error * const
    err = xr.where(err > 0, err, np.nan)
    for R in tmp.R:
        ax.errorbar(
            tmp.t,
            tmp.sel(R=R).values,
            err.sel(R=R).values,
            marker="D",
            alpha=0.9,
            color="red",
        )
    ax.errorbar(
        tmp.t,
        tmp.sel(R=R).values,
        err.sel(R=R).values,
        label="T$_i$(C)",
        marker="D",
        alpha=0.9,
        color="red",
    )

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("(keV)")
    ax.legend()
    ax.set_ylim(bottom=0, top=11)
    ax.set_xlim(tstart, tend)
    if plot_100M:
        plt.hlines(8.6, tstart, tend, linestyle="dashed", color="red", alpha=0.8)
    ax.set_xlabel("Time (s)")
    plt.suptitle(_title)

    if savefig:
        save_figure(fig_name=f"{figname}discharge_evolution")

    # # Ion temperature only a-la PLT PRL
    # rcParams.update({"font.size": 14})
    # plt.figure()
    # ax = plt.axes()
    #
    # const = 1.0e-6
    # nbi = None
    # if "rfx" in raw_data:
    #     nbi = raw_data["rfx"]["pin"] * const
    #     ax.plot(tmp.t, tmp.values, label="P$_{RFX}$", alpha=0.9)
    # if "hnbi1" in raw_data:
    #     tmp = raw_data["hnbi1"]["pin"] * const
    #     if nbi is not None:
    #         nbi += tmp.interp(t=nbi.t)
    #     else:
    #         nbi = tmp
    # if nbi is not None:
    #     ax.plot(nbi.t, nbi.values, label="P$_{NBI}$", alpha=0.9, linestyle="dashed")
    #
    # const = 1.0e-3
    # tmp = raw_data["xrcs"]["te_kw"] * const
    # err = raw_data["xrcs"]["te_kw"].error * const
    # err = xr.where(err > 0, err, np.nan)
    # ax.errorbar(
    #     tmp.t,
    #     tmp.values,
    #     err.values,
    #     label="T$_e$(Ar)",
    #     marker="o",
    #     alpha=0.9,
    # )
    #
    # tmp = raw_data["xrcs"]["ti_w"] * const
    # err = raw_data["xrcs"]["ti_w"].error * const
    # err = xr.where(err > 0, err, np.nan)
    # ax.errorbar(
    #     tmp.t,
    #     tmp.values,
    #     err.values,
    #     label="T$_i$(Ar)",
    #     marker="o",
    #     alpha=0.9,
    # )
    #
    # tmp = raw_data["cxrs"]["ti_full_max"] * const
    # err = raw_data["cxrs"]["ti_full_max"].error * const
    # err = xr.where(err > 0, err, np.nan)
    # for R in tmp.R:
    #     ax.errorbar(
    #         tmp.t,
    #         tmp.sel(R=R).values,
    #         err.sel(R=R).values,
    #         marker="D",
    #         alpha=0.9,
    #         color="red",
    #     )
    # ax.errorbar(
    #     tmp.t,
    #     tmp.sel(R=R).values,
    #     err.sel(R=R).values,
    #     label="T$_i$(C)",
    #     marker="D",
    #     alpha=0.9,
    #     color="red",
    # )
    #
    # ax.set_ylabel("(keV)")
    # ax.legend()
    # ax.set_ylim(bottom=0, top=11)
    # ax.set_xlim(tstart, tend)
    # ax.set_xlabel("Time (s)")
    # ax.set_title(f"Pulse {pulse}")
    # if plot_100M:
    #     plt.hlines(8.6, tstart, tend, linestyle="dashed", color="red", alpha=0.8)
    #
    # if savefig:
    #     save_figure(fig_name=f"{figname}ion_temperature_evolution")

    return st40_data


def save_figure(fig_name="", orientation="landscape", ext=".jpg"):
    _file = "/home/marco.sertoli/figures/Indica/" + fig_name + ext
    plt.savefig(
        _file,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )
    print(f"Saving picture to {_file}")


def get_figname(pulse=None, name=""):

    figname = ""
    if pulse is not None:
        figname = f"{str(int(pulse))}_"

    if len(name) > 0:
        figname += f"{name}_"

    return figname


def plot_10014(savefig=False, run_add_hda="", tplot=0.063):
    import pickle
    import matplotlib.pylab as plt

    plt.ion()

    xrcs = pickle.load(
        open("/home/marco.sertoli/data/profile_stats_XRCS_10014.pkl", "rb")
    )
    xrcs = pickle.load(
        open("/home/marco.sertoli/data/profile_stats_XRCS_nuis_10014.pkl", "rb")
    )
    # xrcs_strahl = pickle.load(
    #     open("/home/marco.sertoli/data/profile_stats_XRCS_strahl_10014.pkl", "rb")
    # )
    cxrs_profs = pickle.load(
        open("/home/marco.sertoli/data/profile_stats_CXRS_10014.pkl", "rb")
    )
    cxrs_data_Ti = pickle.load(
        open("/home/marco.sertoli/data/data_CXRS_ti_10014.pkl", "rb")
    )
    cxrs_data_Vtor = pickle.load(
        open("/home/marco.sertoli/data/data_CXRS_vtor_10014.pkl", "rb")
    )

    import hda.hdatests as tests

    pl_dict, raw_data, data, bckc_dict = tests.read_profile_scans(
        10014, run_add=run_add_hda
    )

    run_plus = 500
    astra_dict = tests.read_profile_scans_astra(13110014, run_plus=run_plus)

    keep = ["60", "64", "65", "73"]
    _pl_dict = {}
    _bckc_dict = {}
    _astra_dict = {}
    for run in keep:
        run_name = f"RUN{run}{run_add_hda}"
        _pl_dict[run] = pl_dict[run_name]
        _bckc_dict[run] = bckc_dict[run_name]

        run_name = f"RUN{int(run) + run_plus}"
        _astra_dict[run] = astra_dict[run_name]

    pl_dict = _pl_dict
    bckc_dict = _bckc_dict
    astra_dict = _astra_dict

    pl = pl_dict[keep[0]]

    st40_data = ST40data(pl.pulse, 0, 0.2)
    st40_data.get_all()
    add_cxrs(st40_data, raw_data)

    Te_all = []
    Ti_all = []
    Ne_all = []
    NAr_all = []
    Nf_all = []
    Ni_all = []
    te_xrcs_bckc = []
    ti_xrcs_bckc = []
    rho_mean_te = []
    rho_in_te = []
    rho_out_te = []
    rho_mean_ti = []
    rho_in_ti = []
    rho_out_ti = []
    runs = list(pl_dict)
    for run in keep:
        pl = pl_dict[run]
        astra = astra_dict[run]

        Te_all.append(pl.el_temp.sel(t=tplot, method="nearest"))
        Ti_all.append(pl.ion_temp.sel(element="ar").sel(t=tplot, method="nearest"))
        Ne_all.append(pl.el_dens.sel(t=tplot, method="nearest"))
        NAr_all.append(pl.ion_dens.sel(element="ar").sel(t=tplot, method="nearest"))
        Nf_all.append(
            (astra["nf"] * 1.0e19)
            .interp(t=tplot, method="nearest")
            .interp(rho_poloidal=pl.rho)
        )
        Ni_all.append(
            (astra["ni"] * 1.0e19)
            .interp(t=tplot, method="nearest")
            .interp(rho_poloidal=pl.rho)
        )

        te_tmp = bckc_dict[run]["xrcs"]["te_n3w"].sel(t=tplot, method="nearest")
        rho_tmp = te_tmp.pos.sel(t=tplot, method="nearest")
        te_xrcs_bckc.append(te_tmp)
        rho_mean_te.append(te_tmp.pos.sel(t=tplot, method="nearest").value)
        rho_in_te.append(rho_tmp.value - rho_tmp.err_in)
        rho_out_te.append(rho_tmp.value + rho_tmp.err_out)

        ti_tmp = bckc_dict[run]["xrcs"]["ti_w"].sel(t=tplot, method="nearest")
        rho_tmp = ti_tmp.pos.sel(t=tplot, method="nearest")
        ti_xrcs_bckc.append(ti_tmp)
        rho_mean_ti.append(rho_tmp.value)
        rho_in_ti.append(rho_tmp.value - rho_tmp.err_in)
        rho_out_ti.append(rho_tmp.value + rho_tmp.err_out)

    Te_all = xr.concat(Te_all, "run").assign_coords({"run": runs})
    Ti_all = xr.concat(Ti_all, "run").assign_coords({"run": runs})
    Ne_all = xr.concat(Ne_all, "run").assign_coords({"run": runs})
    NAr_all = xr.concat(NAr_all, "run").assign_coords({"run": runs})
    Nf_all = xr.concat(Nf_all, "run").assign_coords({"run": runs})
    Ni_all = xr.concat(Ni_all, "run").assign_coords({"run": runs})
    te_xrcs_bckc = xr.concat(te_xrcs_bckc, "run").assign_coords({"run": runs})
    rho_mean_te = xr.concat(rho_mean_te, "run").assign_coords({"run": runs})
    rho_out_te = xr.concat(rho_out_te, "run").assign_coords({"run": runs})
    rho_in_te = xr.concat(rho_in_te, "run").assign_coords({"run": runs})
    ti_xrcs_bckc = xr.concat(ti_xrcs_bckc, "run").assign_coords({"run": runs})
    rho_mean_ti = xr.concat(rho_mean_ti, "run").assign_coords({"run": runs})
    rho_out_ti = xr.concat(rho_out_ti, "run").assign_coords({"run": runs})
    rho_in_ti = xr.concat(rho_in_ti, "run").assign_coords({"run": runs})

    # R_shift_cxrs = 0.04
    # ti_cxrs_full = (
    #     raw_data["cxrs"]["ti_full"].sel(t=tplot, method="nearest").drop_vars("t")
    # )
    # ti_cxrs_full.attrs["error"] = ti_cxrs_full.attrs["error"].sel(t=tplot, method="nearest").drop_vars("t")
    # ti_cxrs_bgnd = (
    #     raw_data["cxrs"]["ti_bgnd"].sel(t=tplot, method="nearest").drop_vars("t")
    # )
    # ti_cxrs_bgnd.attrs["error"] = ti_cxrs_bgnd.attrs["error"].sel(t=tplot, method="nearest").drop_vars("t")

    R_shift_cxrs = 0.02
    cxrs_data = pickle.load(
        open("/home/marco.sertoli/data/data_CXRS_ti_10014.pkl", "rb")
    )
    ti_cxrs_full = DataArray(
        cxrs_data["data"]["ff"],
        coords=[("R", cxrs_data["data"]["ff_R"])],
        attrs={
            "error": DataArray(
                cxrs_data["data"]["ff_err"], coords=[("R", cxrs_data["data"]["ff_R"])]
            )
        },
    )
    ti_cxrs_bgnd = DataArray(
        cxrs_data["data"]["bs"],
        coords=[("R", cxrs_data["data"]["bs_R"])],
        attrs={
            "error": DataArray(
                cxrs_data["data"]["bs_err"], coords=[("R", cxrs_data["data"]["bs_R"])]
            )
        },
    )

    equilibrium = pl.equilibrium
    channel = np.arange(len(ti_cxrs_full.R))
    cxrs_R = (
        ti_cxrs_full.R.assign_coords(channel=("R", channel))
        .swap_dims({"R": "channel"})
        .drop_vars("R")
    ) + R_shift_cxrs
    cxrs_z = xr.full_like(cxrs_R, 0)
    cxrs_rho = equilibrium.rho.interp(t=tplot, method="nearest").drop_vars("t")
    cxrs_rho = cxrs_rho.interp(R=cxrs_R, z=cxrs_z)
    ti_cxrs_full = ti_cxrs_full.assign_coords(rho_poloidal=("R", cxrs_rho))
    ti_cxrs_bgnd = ti_cxrs_bgnd.assign_coords(rho_poloidal=("R", cxrs_rho))

    const = 1.0e-19
    plt.figure()
    mean = Ne_all.mean("run") * const
    up = Ne_all.max("run") * const
    low = Ne_all.min("run") * const
    plt.plot(mean.rho_poloidal, mean, color="blue", label="Electrons")
    plt.fill_between(mean.rho_poloidal, up, low, alpha=0.5, color="blue")

    mean = Ni_all.mean("run") * const
    up = Ni_all.max("run") * const
    low = Ni_all.min("run") * const
    plt.plot(mean.rho_poloidal, mean, color="red", label="Thermal ions")
    plt.fill_between(mean.rho_poloidal, up, low, alpha=0.5, color="red")

    mean = Nf_all.mean("run") * const
    up = Nf_all.max("run") * const
    low = Nf_all.min("run") * const
    plt.plot(mean.rho_poloidal, mean, color="green", label="Fast ions")
    plt.fill_between(mean.rho_poloidal, up, low, alpha=0.5, color="green")
    plt.title("Densities")
    plt.legend()
    plt.xlabel(r"$\rho_{pol}$")
    plt.ylabel("($10^{19}$ $m^{-3}$)")
    if savefig:
        save_figure(fig_name=f"10014_el_and_ion_densities")

    const = 1.0e-16
    plt.figure()
    mean = NAr_all.mean("run") * const
    up = NAr_all.max("run") * const
    low = NAr_all.min("run") * const
    plt.plot(mean.rho_poloidal, mean, color="orange", label="Argon")
    plt.fill_between(mean.rho_poloidal, up, low, alpha=0.5, color="orange")
    plt.title("Impurity density")
    plt.legend()
    plt.xlabel(r"$\rho_{pol}$")
    plt.ylabel("($10^{16}$ $m^{-3}$)")
    if savefig:
        save_figure(fig_name=f"10014_argon_density")

    const = 1.0e-3
    plt.figure()
    mean = Te_all.mean("run") * const
    up = Te_all.max("run") * const
    low = Te_all.min("run") * const
    plt.plot(mean.rho_poloidal, mean, color="blue", label="Electrons")
    plt.fill_between(mean.rho_poloidal, up, low, alpha=0.5, color="blue")
    plt.errorbar(
        rho_mean_te.mean("run"),
        te_xrcs_bckc.mean("run") * const,
        (te_xrcs_bckc.min("run") - te_xrcs_bckc.max("run")) * const,
        marker="o",
        color="blue",
    )
    plt.hlines(
        te_xrcs_bckc.mean("run") * const,
        rho_in_te.min("run"),
        rho_out_te.max("run"),
        color="blue",
    )

    mean = Ti_all.mean("run") * const
    up = Ti_all.max("run") * const
    low = Ti_all.min("run") * const
    plt.plot(mean.rho_poloidal, mean, color="red", label="Ions")
    plt.fill_between(mean.rho_poloidal, up, low, alpha=0.5, color="red")
    plt.errorbar(
        rho_mean_ti.mean("run"),
        ti_xrcs_bckc.mean("run") * const,
        (ti_xrcs_bckc.min("run") - ti_xrcs_bckc.max("run")) * const,
        marker="o",
        color="red",
        label="XRCS",
    )
    plt.hlines(
        ti_xrcs_bckc.mean("run") * const,
        rho_in_ti.min("run"),
        rho_out_ti.max("run"),
        color="red",
    )
    plt.errorbar(
        ti_cxrs_bgnd.rho_poloidal,
        xr.where(ti_cxrs_bgnd > 0, ti_cxrs_bgnd * const, np.nan),
        ti_cxrs_bgnd.error * const,
        marker="s",
        mfc="white",
        color="red",
        label="CXRS bgnd-subtr",
        linestyle="",
    )
    plt.errorbar(
        ti_cxrs_full.rho_poloidal,
        xr.where(ti_cxrs_full > 0, ti_cxrs_full * const, np.nan),
        ti_cxrs_full.error * const,
        marker="^",
        mfc="white",
        color="red",
        label="CXRS full-fit",
        linestyle="",
    )
    plt.title("Temperatures")
    plt.legend(fontsize=10)
    plt.xlabel(r"$\rho_{pol}$")
    plt.ylabel("(keV)")
    if savefig:
        save_figure(fig_name=f"10014_temperatures")

    return pl_dict, data, raw_data, bckc_dict

    plt.figure()
    cols = cmap(np.linspace(0, 1, len(ind_best)))
    for i in range(0, len(cols), 10):
        val = xrcs["raw_profiles"][ind_best[i]]["Ne"] ** 2
        val.plot(color=cols[i], alpha=0.3)
    val.plot(color=cols[i], alpha=0.3, label="Ne**2")

    cols = cmap(np.linspace(0, 1, len(runs)))
    for run, col in zip(runs, cols):
        pl = pl_dict[run]
        ne_nimp = pl.el_dens.sel(t=t, method="nearest") * pl.ion_dens.sel(
            element="ar"
        ).sel(t=t, method="nearest")
        (ne_nimp / ne_nimp.max() * 1.0e19 * 1.0e19).plot(
            color=col,
            label=run,
            linestyle="dashed",
            linewidth=3,
        )
    plt.title("Ne * NAr")
    plt.legend(fontsize=9)

    keys = ["Ti", "Te"]
    for k in keys:
        plt.figure()
        cols = cmap(np.linspace(0, 1, len(ind_best)))
        for i in range(0, len(cols), 10):
            val = xrcs["raw_profiles"][ind_best[i]][k]
            val.plot(color=cols[i], alpha=0.3)
        val.plot(color=cols[i], alpha=0.3, label=k)

        cols = cmap(np.linspace(0, 1, len(runs)))
        for run, col in zip(runs, cols):
            pl = pl_dict[run]
            if k == "Te":
                Te = pl.el_temp.sel(t=t, method="nearest")
                Te.plot(
                    color=col,
                    label=run,
                    linestyle="dashed",
                    linewidth=3,
                )
                if best_hda in run and hasattr(pl, "el_temp_hi"):
                    Te_err = (
                        pl.el_temp_hi.sel(t=t, method="nearest")
                        - pl.el_temp_lo.sel(t=t, method="nearest")
                    ) / 2.0
                    plt.fill_between(
                        pl.rho,
                        Te - Te_err,
                        Te + Te_err,
                        color=col,
                        alpha=0.8,
                    )
            if k == "Ti":
                Ti = pl.ion_temp.sel(element="ar").sel(t=t, method="nearest")
                Ti.plot(
                    color=col,
                    label=run,
                    linestyle="dashed",
                    linewidth=3,
                )
                if best_hda in run and hasattr(pl, "el_temp_hi"):
                    Ti_err = (
                        pl.ion_temp_hi.sel(element="ar").sel(t=t, method="nearest")
                        - pl.ion_temp_lo.sel(element="ar").sel(t=t, method="nearest")
                    ) / 2.0
                    plt.fill_between(
                        pl.rho,
                        Ti - Ti_err,
                        Ti + Ti_err,
                        color=col,
                        alpha=0.8,
                    )

        plt.title(k)
        plt.legend(fontsize=9)

    return pl_dict, data, raw_data, bckc_dict

    xrcs_keys = ["te_kw", "te_n3w"]
    data["xrcs"]["te_avrg"] = xr.full_like(data["xrcs"]["te_kw"], np.nan)
    data["xrcs"]["te_avrg"].attrs["error"] = xr.full_like(
        data["xrcs"]["te_kw"].error, np.nan
    )
    data["xrcs"]["te_avrg"].name = "xrcs_te_avrg"
    for t in data["xrcs"]["te_avrg"].t:
        val = []
        err = []
        for k in xrcs_keys:
            _val = data["xrcs"][k].sel(t=t)
            if np.isfinite(_val):
                val.append(_val)
            _err = data["xrcs"][k].error.sel(t=t)
            if np.isfinite(_err):
                err.append(_err)
        if len(val) > 0:
            data["xrcs"]["te_avrg"].loc[dict(t=t)] = np.sum(val) / len(val)
        if len(err) > 0:
            err_tmp = np.sqrt(
                np.sum((np.array(err) / len(err)) ** 2 + np.std(val) ** 2)
            )
            data["xrcs"]["te_avrg"].attrs["error"].loc[dict(t=t)] = err_tmp

    data["xrcs"]["te_kw"] = data["xrcs"]["te_avrg"]
    data["xrcs"]["te_n3w"] = data["xrcs"]["te_avrg"]
    data["xrcs"]["te_n3w"][0] = data["xrcs"]["te_n3w"][1]
    data["xrcs"]["te_kw"][0] = data["xrcs"]["te_kw"][1]
    data["xrcs"]["ti_w"][0] = data["xrcs"]["ti_w"][1]
    data["xrcs"]["int_n3/int_w"][0] = data["xrcs"]["int_n3/int_w"][1]
    data["xrcs"]["int_n3/int_tot"][0] = data["xrcs"]["int_n3/int_tot"][1]
    data["xrcs"]["int_k/int_w"][0] = data["xrcs"]["int_k/int_w"][1]

    plt.figure()
    const = 1.0
    plt.errorbar(
        xrcs["data"].wavelength * 10,
        xrcs["data"] * const,
        xrcs["data_err"] * const,
        label="Data",
        marker="o",
        markersize=3,
        color="gray",
        zorder=-1,
    )
    plt.fill_between(
        xrcs["model"]["mean"].wavelength * 10,
        xrcs["model"]["lower"] * const,
        xrcs["model"]["upper"] * const,
        color="r",
        alpha=0.8,
        label="Model",
    )
    # plt.fill_between(
    #     xrcs_strahl["model"]["mean"].wavelength * 10,
    #     xrcs_strahl["model"]["lower"] * const,
    #     xrcs_strahl["model"]["upper"] * const,
    #     color="r",
    #     alpha=0.8,
    # )
    plt.xlim(3.943, 3.962)
    plt.legend()
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.xticks([3.9450, 3.950, 3.955, 3.960])
    plt.ylabel("Intensity (a.u.)")
    plt.title("He-like Ar spectra")
    if savefig:
        save_figure(fig_name=f"10014_XRCS_spectra")

    plt.figure()
    const = 1.0e-3
    ind = np.where(cxrs_data_Ti["data"]["ff"] > 0)[0]
    ff = cxrs_data_Ti["data"]["ff"][ind] * const
    ff_err = cxrs_data_Ti["data"]["ff_err"][ind] * const
    ff_R = cxrs_data_Ti["data"]["ff_R"][ind]
    plt.errorbar(
        ff_R,
        ff,
        ff_err,
        label="Data (full-fit)",
        marker="o",
        markersize=5,
        color="black",
        linestyle="",
    )
    ind = np.where(cxrs_data_Ti["data"]["bs"] > 0)[0]
    bs = cxrs_data_Ti["data"]["bs"][ind] * const
    bs_err = cxrs_data_Ti["data"]["bs_err"][ind] * const
    bs_R = cxrs_data_Ti["data"]["bs_R"][ind]
    plt.errorbar(
        bs_R,
        bs,
        bs_err,
        label="Data (bgnd-subtr)",
        marker="^",
        markersize=5,
        color="gray",
        linestyle="",
    )
    ind = np.where(
        (cxrs_data_Ti["model"]["mean"].R >= np.append(bs_R, ff_R).min() * 0.9)
        * (cxrs_data_Ti["model"]["mean"].R <= np.append(bs_R, ff_R).max() * 1.1)
    )[0]
    plt.fill_between(
        cxrs_data_Ti["model"]["mean"].R[ind],
        cxrs_data_Ti["model"]["lower"][ind] * const,
        cxrs_data_Ti["model"]["upper"][ind] * const,
        color="r",
        alpha=0.8,
        label="Model",
    )
    plt.xticks([0.45, 0.50, 0.55, 0.60])
    plt.legend()
    plt.xlabel(r"R (m)")
    plt.ylabel("(keV)")
    plt.title("CXRS Ion temperature")
    if savefig:
        save_figure(fig_name=f"10014_CXRS_ion_temperature_data")

    plt.figure()
    const = 1.0e-3
    ind = np.where(cxrs_data_Vtor["data"]["ff"] > 0)[0]
    ff = cxrs_data_Vtor["data"]["ff"][ind] * const
    ff_err = cxrs_data_Vtor["data"]["ff_err"][ind] * const
    ff_R = cxrs_data_Vtor["data"]["ff_R"][ind]
    plt.errorbar(
        ff_R,
        ff,
        ff_err,
        label="Data (full-fit)",
        marker="o",
        markersize=5,
        color="black",
        linestyle="",
    )
    ind = np.where(cxrs_data_Vtor["data"]["bs"] > 0)[0]
    bs = cxrs_data_Vtor["data"]["bs"][ind] * const
    bs_err = cxrs_data_Vtor["data"]["bs_err"][ind] * const
    bs_R = cxrs_data_Vtor["data"]["bs_R"][ind]
    plt.errorbar(
        bs_R,
        bs,
        bs_err,
        label="Data (bgnd-subtr)",
        marker="^",
        markersize=5,
        color="gray",
        linestyle="",
    )
    ind = np.where(
        (cxrs_data_Vtor["model"]["mean"].R >= np.append(bs_R, ff_R).min() * 0.9)
        * (cxrs_data_Vtor["model"]["mean"].R <= np.append(bs_R, ff_R).max() * 1.1)
    )[0]
    plt.fill_between(
        cxrs_data_Vtor["model"]["mean"].R[ind],
        -cxrs_data_Vtor["model"]["lower"][ind],
        -cxrs_data_Vtor["model"]["upper"][ind],
        color="r",
        alpha=0.8,
        label="Model",
    )
    plt.xticks([0.45, 0.50, 0.55, 0.60])
    plt.legend()
    plt.xlabel(r"R (m)")
    plt.ylabel("(krad/s)")
    plt.title("CXRS Toroidal rotation")
    if savefig:
        save_figure(fig_name=f"10014_CXRS_toroidal_rotation_data")

    plt.figure()
    const = 1.0e-3
    rho = np.linspace(0, 1, 41)
    # lower = xrcs["Ti"]["lower"].interp(rho_poloidal=rho)*const
    # upper = xrcs["Ti"]["upper"].interp(rho_poloidal=rho)*const
    # plt.fill_between(
    #     rho,
    #     lower,
    #     upper,
    #     color="orange",
    #     alpha=0.8,
    #     label="Ti(XRCS)"
    # )
    lower = xrcs["Te"]["lower"].interp(rho_poloidal=rho) * const
    upper = xrcs["Te"]["upper"].interp(rho_poloidal=rho) * const
    plt.fill_between(rho, lower, upper, color="r", alpha=0.8, label="Electron")
    plt.fill_between(
        cxrs_profs["Ti"]["mean"].rho_poloidal,
        cxrs_profs["Ti"]["lower"] * const,
        cxrs_profs["Ti"]["upper"] * const,
        color="b",
        alpha=0.5,
        label="Impurity (C/Ar)",
    )
    plt.legend()
    plt.title("Temperatures")
    plt.xlabel(r"$\rho_{pol}$")
    plt.ylabel("(keV)")
    if savefig:
        save_figure(fig_name=f"10014_temperatures")

    plt.figure()
    const = 1.0e-3
    rho = np.linspace(0, 1, 41)
    plt.fill_between(
        cxrs_profs["Vtor"]["mean"].rho_poloidal,
        cxrs_profs["Vtor"]["lower"] * const,
        cxrs_profs["Vtor"]["upper"] * const,
        color="b",
        alpha=0.5,
        label="Impurity (C/Ar)",
    )
    plt.legend()
    plt.title("Toroidal rotation")
    plt.xlabel(r"$\rho_{pol}$")
    plt.ylabel("(krad/s)")
    if savefig:
        save_figure(fig_name=f"10014_rotation")

    plt.figure()
    const = 1.0e-19
    rho = np.linspace(0, 1, 41)
    lower = xrcs["Ne"]["lower"].interp(rho_poloidal=rho) * const
    upper = xrcs["Ne"]["upper"].interp(rho_poloidal=rho) * const
    plt.fill_between(rho, lower, upper, color="r", alpha=0.8, label="Electron")
    # lower = xrcs_strahl["Ne"]["lower"].interp(rho_poloidal=rho) * const
    # upper = xrcs_strahl["Ne"]["upper"].interp(rho_poloidal=rho) * const
    # plt.fill_between(rho, lower, upper, color="r", alpha=0.8)
    plt.legend()
    plt.title("Densities")
    plt.xlabel(r"$\rho_{pol}$")
    plt.ylabel("($10^{19}$ $m^{-3}$)")
    if savefig:
        save_figure(fig_name=f"10014_densities")

    return xrcs, cxrs_profs
