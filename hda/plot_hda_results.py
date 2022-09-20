import matplotlib.pylab as plt
import numpy as np
from matplotlib import rcParams, cm
from xarray import DataArray
import xarray as xr
from copy import deepcopy
from scipy import constants

from indica.readers import ST40Reader
from hda.read_st40 import ST40data
import pickle

plt.ion()

cxrs_analyses = {"ff": "full-fit", "bs": "bgnd-subtr."}
cxrs_markers = {"ff": "s", "bs": "^"}
default_marker = "o"
xrcs_marker = "o"
smmh1_marker = ""
const_nTtaue = 1.0e-3 * 1.0e-18
const_taue = 1.0e3
const_power = 1.0e-6
const_weq = 1.0e-3
const_temp = 1.0e-3
const_dens = 1.0e-19
const_imp = 1.0e-16
const_rot = 1.0e-3
label_nTtaue = "($10^{18} m^{-3}$ keV s)"
label_taue = "(ms)"
label_power = "(MW)"
label_wp = "(kJ)"
label_temp = "(eV)"
label_dens = "($10^{19}$ $m^{-3}$)"
label_time = "Time (s)"
alpha = 0.8
tlim = (0.02,)


def set_sizes(fontsize=16, legendsize=14, markersize=9):
    rcParams.update({"font.size": fontsize})
    rcParams.update({"font.size": fontsize})
    rcParams.update({"legend.fontsize": legendsize})
    rcParams.update({"lines.markersize": markersize})


def save_figure(fig_name="", orientation="landscape", ext="jpg"):
    _file = f"/home/marco.sertoli/figures/Indica/{fig_name}.{ext}"
    plt.savefig(
        _file, orientation=orientation, dpi=600, pil_kwargs={"quality": 95},
    )
    print(f"Saving picture to {_file}")


def get_figname(pulse=None, name=""):

    figname = ""
    if pulse is not None:
        figname = f"{str(int(pulse))}_"

    if len(name) > 0:
        figname += f"{name}_"

    return figname


def add_cxrs(st40_data: ST40data, raw_data: dict, R_shift=0.02):

    raw_data["cxrs"] = {}

    t_slice = slice(st40_data.tstart, st40_data.tend)
    if st40_data.pulse == 10014:
        t_slice = slice(0.04, st40_data.tend)

    # Geometry correction (status @ 04/07/22)
    channels = [22, 23, 24, 25, 26]
    R_corrected = np.array([0.6189, 0.5398, 0.48494, 0.4449, 0.4211]) + R_shift
    angle_correction = [1.0072, 1.0072, 1.0072, 1.0072, 1.0072]

    # RUN numbers: ff=full-fit, bs=background-subtraction (status @ 04/07/2022)
    pulse_info = {
        10014: {"ff": 7, "bs": 9},
        10009: {"ff": 3, "bs": 1},
        9831: {"ff": 5, "bs": 5},
        9780: {"ff": 5, "bs": 6},
    }

    # Restrict data to channels 22-26
    info = pulse_info[st40_data.pulse]
    for analysis, rev in info.items():
        ti, dims = st40_data.reader._get_data(
            "spectrom", "princeton.cxsfit_out", ":ti", rev
        )
        ti_err, dims = st40_data.reader._get_data(
            "spectrom", "princeton.cxsfit_out", ":ti_err", rev
        )
        print("\n Shifting time dimension until hard-coded in MDS+!!! \n")
        time = dims[1]
        dt = time[1] - time[0]
        time_mid = time + dt / 2
        ti = (
            DataArray(ti[:, :5], coords=[("t", time_mid), ("R", R_corrected)])
            .sel(t=t_slice)
            .sortby("R")
        )
        ti_err = (
            DataArray(ti_err[:, :5], coords=[("t", time_mid), ("R", R_corrected)])
            .sel(t=t_slice)
            .sortby("R")
        )
        ti.attrs = {"error": ti_err}
        raw_data["cxrs"][f"ti_{analysis}"] = ti

        vtor, dims = st40_data.reader._get_data(
            "spectrom", "princeton.cxsfit_out", ":vtor", rev
        )
        vtor_err, dims = st40_data.reader._get_data(
            "spectrom", "princeton.cxsfit_out", ":vtor_err", rev
        )
        time = dims[1]
        dt = time[1] - time[0]
        time_mid = time + dt / 2
        vtor = (
            DataArray(
                vtor[:, :5] * angle_correction,
                coords=[("t", time_mid), ("R", R_corrected)],
            )
            .sel(t=t_slice)
            .sortby("R")
        )
        vtor_err = (
            DataArray(vtor_err[:, :5], coords=[("t", time_mid), ("R", R_corrected)])
            .sel(t=t_slice)
            .sortby("R")
        )
        vtor.attrs = {"error": vtor_err}
        raw_data["cxrs"][f"vtor_{analysis}"] = vtor


def load_xrcs():
    xrcs = pickle.load(
        open("/home/marco.sertoli/data/profile_stats_XRCS_nuis_10014.pkl", "rb")
    )
    return xrcs


def load_pickle_HDA(pulse: int, name: str, path="/home/marco.sertoli/data/Indica/"):
    picklefile = f"{path}{pulse}_{name}_HDA.pkl"
    return pickle.load(open(picklefile, "rb"))


def read_profile_scans_HDA(pulse, run_add=""):
    runs = np.arange(60, 76 + 1)
    pl_dict = {}
    bckc_dict = {}
    for run in runs:
        run_name = f"RUN{run}{run_add}"
        pl, raw_data, data, bckc = load_pickle_HDA(pulse, run_name)
        pl_dict[run_name] = deepcopy(pl)
        bckc_dict[run_name] = deepcopy(bckc)

    return pl_dict, raw_data, data, bckc_dict


def read_profile_scans_ASTRA(pulse, run_add="", run_plus=500):
    runs = np.arange(60, 76 + 1)
    astra_dict = {}
    reader_astra = ST40Reader(pulse, 0, 0.2, tree="ASTRA")
    for run in runs:
        revision = f"{run+run_plus}{run_add}"
        run_name = f"RUN{revision}"
        astra_dict[run_name] = reader_astra.get("", "astra", revision)

        Pe = astra_dict[run_name]["ne"] * astra_dict[run_name]["te"]
        Pe = Pe * constants.e * 1.0e3 * 1.0e19
        Pi = astra_dict[run_name]["ni"] * astra_dict[run_name]["ti"]
        Pi = Pi * constants.e * 1.0e3 * 1.0e19
        Pth = Pe + Pi
        Pblon = astra_dict[run_name]["pblon"]
        Pbperp = astra_dict[run_name]["pbper"]
        volume = astra_dict[run_name]["volume"]

        astra_dict[run_name]["wastra"] = deepcopy(astra_dict[run_name]["wth"])
        astra_dict[run_name]["wastra"].name = "astra_stored_energy"
        astra_dict[run_name]["wastra"].attrs["datatype"] = ("stored_energy", "astra")

        astra_dict[run_name]["wth"].name = "thermal_stored_energy"
        astra_dict[run_name]["wth"].attrs["datatype"] = ("stored_energy", "thermal")

        astra_dict[run_name]["weq"] = xr.zeros_like(astra_dict[run_name]["wth"])
        astra_dict[run_name]["weq"].name = "equilibrium_stored_energy"
        astra_dict[run_name]["weq"].attrs["datatype"] = ("stored_energy", "equilibrium")

        astra_dict[run_name]["wtot"] = xr.zeros_like(astra_dict[run_name]["wth"])
        astra_dict[run_name]["wtot"].name = "total_stored_energy"
        astra_dict[run_name]["wtot"].attrs["datatype"] = ("stored_energy", "total")
        for t in astra_dict[run_name]["wth"].t:
            vol_tmp = volume.sel(t=t, method="nearest")

            wth = 3 / 2 * np.trapz(Pth.sel(t=t), vol_tmp)
            weq = wth + np.trapz(3 / 4 * (Pbperp + Pblon).sel(t=t), vol_tmp)
            wtot = wth + np.trapz((1 / 2 * Pblon + Pbperp).sel(t=t), vol_tmp)

            astra_dict[run_name]["wth"].loc[dict(t=t)] = wth
            astra_dict[run_name]["weq"].loc[dict(t=t)] = weq
            astra_dict[run_name]["wtot"].loc[dict(t=t)] = wtot

    return astra_dict


def read_fidasim_results():

    ti_fidasim = pickle.load(
        open("/home/marco.sertoli/data/data_CXRS_ti_10014.pkl", "rb")
    )
    vtor_fidasim = pickle.load(
        open("/home/marco.sertoli/data/data_CXRS_vtor_10014.pkl", "rb")
    )

    return ti_fidasim, vtor_fidasim


def plot(
    pulse,
    savefig=False,
    plot_all=False,
    use_std=False,
    use_std_ti=True,
    use_std_vtor=True,
    plot_bckc=False,
    tlim=(0.02, 0.1),
    ext="jpg",
    multiplot=False,
):
    set_sizes()

    all_runs = [str(run) for run in np.arange(60, 76 + 1)]

    if pulse == 10014:
        tplot = 0.0675
        # tplot = 0.064
        keep = ["60", "64", "65", "73"]
        keep = all_runs[:5]
        run_plus_astra = 500
        run_add_hda = "MID"
        omega_scaling = 420e3
        R_shift = 0.02
    elif pulse == 10009:
        tplot = 0.058
        # keep = ["60", "64", "65", "73"]  # initial best runs sent to Stan
        keep = ["60", "64", "65", "66", "72", "73"]
        run_plus_astra = 500
        run_add_hda = "MID"
        omega_scaling = 440e3
        R_shift = 0.03
        bckc_R = np.array([0.62, 0.54, 0.48, 0.44, 0.42]) + R_shift
        bckc_vtor = [
            np.array([44, 110, 177, 117, 74]) / bckc_R * 1.0e3,
            # np.array([29, 86, 170, 104, 59]) / bckc_R * 1.0e3,
            np.array([52, 126, 182, 128, 85]) / bckc_R * 1.0e3,
            np.array([43, 113, 175, 118, 74]) / bckc_R * 1.0e3,
        ]

        bckc_ti = [
            np.array([2119, 5420, 9328, 7228, 5138]),
            # np.array([1334, 4286, 9125, 6526, 4068]),
            np.array([2355, 5833, 8859, 7329, 5428]),
            np.array([2217, 6102, 10216, 8291, 5840]),
        ]
    elif pulse == 9831:
        tplot = 0.076
        keep = ["60", "71", "73"]
        run_plus_astra = 500
        run_add_hda = ""
        omega_scaling = 450e3
        R_shift = 0.02
    elif pulse == 9780:
        tplot = 0.084
        keep = ["60", "71", "73"]
        run_plus_astra = 500
        run_add_hda = "MID"
        omega_scaling = 550e3
        R_shift = 0.0
    else:
        raise ValueError(f"...input missing for pulse {pulse}..")

    cmap = cm.rainbow
    varr = np.linspace(0, 1, len(keep))
    colors = cmap(varr)

    astra_pulse = int(pulse + 13.1e6)

    pl_dict_all, raw_data, data, bckc_dict_all = read_profile_scans_HDA(
        pulse, run_add=run_add_hda
    )
    astra_dict_all = read_profile_scans_ASTRA(astra_pulse, run_plus=run_plus_astra)

    for run in all_runs:
        run_name = f"RUN{run}{run_add_hda}"
        pl_dict_all[run] = pl_dict_all[run_name]
        bckc_dict_all[run] = bckc_dict_all[run_name]
        if run != run_name:
            pl_dict_all.pop(run_name)
            bckc_dict_all.pop(run_name)

        run_name = f"RUN{int(run) + run_plus_astra}"
        astra_dict_all[run] = astra_dict_all[run_name]
        if run != run_name:
            astra_dict_all.pop(run_name)

    pl_dict = {}
    bckc_dict = {}
    astra_dict = {}
    for run in keep:
        pl_dict[run] = pl_dict_all[run]
        bckc_dict[run] = bckc_dict_all[run]
        astra_dict[run] = astra_dict_all[run]

    st40_data = ST40data(pulse, tlim[0], tlim[1])
    st40_data.get_all()
    add_cxrs(st40_data, raw_data, R_shift=R_shift)

    P_oh_all = []
    Pnb_all = []
    Pabs_all = []
    Ptot_all = []
    Te_all = []
    Ti_all = []
    Ne_all = []
    NAr_all = []
    Nf_all = []
    Ni_all = []
    Weq_all = []
    Wth_all = []
    Wastra_all = []
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
        Weq_all.append(astra["weq"])
        Wastra_all.append(astra["wastra"])
        Wth_all.append(astra["wtherm"])
        P_oh_all.append(astra["p_oh"])
        Pabs_all.append(astra["pabs"])
        Pnb_all.append(astra["pnb"])
        Ptot_all.append(astra["pabs"] + astra["p_oh"])

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
    Weq_all = xr.concat(Weq_all, "run").assign_coords({"run": runs})
    Wth_all = xr.concat(Wth_all, "run").assign_coords({"run": runs})
    Wastra_all = xr.concat(Wastra_all, "run").assign_coords({"run": runs})
    P_oh_all = xr.concat(P_oh_all, "run").assign_coords({"run": runs})
    Ptot_all = xr.concat(Ptot_all, "run").assign_coords({"run": runs})
    Pabs_all = xr.concat(Pabs_all, "run").assign_coords({"run": runs})
    Pnb_all = xr.concat(Pnb_all, "run").assign_coords({"run": runs})
    Ptot_all = xr.concat(Ptot_all, "run").assign_coords({"run": runs})

    te_xrcs_bckc = xr.concat(te_xrcs_bckc, "run").assign_coords({"run": runs})
    rho_mean_te = xr.concat(rho_mean_te, "run").assign_coords({"run": runs})
    rho_out_te = xr.concat(rho_out_te, "run").assign_coords({"run": runs})
    rho_in_te = xr.concat(rho_in_te, "run").assign_coords({"run": runs})
    ti_xrcs_bckc = xr.concat(ti_xrcs_bckc, "run").assign_coords({"run": runs})
    rho_mean_ti = xr.concat(rho_mean_ti, "run").assign_coords({"run": runs})
    rho_out_ti = xr.concat(rho_out_ti, "run").assign_coords({"run": runs})
    rho_in_ti = xr.concat(rho_in_ti, "run").assign_coords({"run": runs})

    # Calculate rho of PI LOS-beam intersection and add infor to data
    cxrs = deepcopy(raw_data["cxrs"])
    equilibrium = pl.equilibrium
    bckc_rho = equilibrium.rho.interp(t=tplot, method="nearest").drop_vars("t")
    bckc_rho = bckc_rho.interp(R=bckc_R, z=bckc_R * 0.0)[0].values
    for analysis_key, analysis in cxrs_analyses.items():
        for quantity in ["ti", "vtor"]:
            key = f"{quantity}_{analysis_key}"
            cxrs[key].name = analysis
            cxrs[key] = cxrs[key].assign_coords(z=("R", xr.full_like(cxrs[key].R, 0)))

            cxrs_rho = equilibrium.rho.interp(t=tplot, method="nearest").drop_vars("t")
            cxrs_rho = cxrs_rho.interp(R=cxrs[key].R, z=cxrs[key].z)
            cxrs[key] = cxrs[key].assign_coords(rho_poloidal=("R", cxrs_rho))

    # Time evolution
    plt.figure()
    value = raw_data["cxrs"]["ti_ff"][:, 2].sel(t=slice(tlim[0], tlim[1]))
    error = raw_data["cxrs"]["ti_ff"].error[:, 2].sel(t=slice(tlim[0], tlim[1]))
    value = xr.where(value > 0, value, np.nan)
    error = xr.where(value > 0, error, np.nan)
    plt.errorbar(
        value.t,
        value * const_temp,
        error * const_temp,
        marker=cxrs_markers["ff"],
        color="red",
        alpha=alpha,
        label="Ti CXRS",
    )

    value = raw_data["cxrs"]["ti_bs"][:, 2].sel(t=slice(tlim[0], tlim[1]))
    error = raw_data["cxrs"]["ti_bs"].error[:, 2].sel(t=slice(tlim[0], tlim[1]))
    value = xr.where(value > 0, value, np.nan)
    error = xr.where(value > 0, error, np.nan)
    plt.errorbar(
        value.t,
        value * const_temp,
        error * const_temp,
        marker=cxrs_markers["bs"],
        color="red",
        alpha=alpha,
        label="Ti CXRS (bgnd-subtr)",
    )

    value = raw_data["xrcs"]["ti_w"].sel(t=slice(tlim[0], tlim[1]))
    error = raw_data["xrcs"]["ti_w"].error.sel(t=slice(tlim[0], tlim[1]))
    plt.errorbar(
        value.t,
        value * const_temp,
        error * const_temp,
        marker=xrcs_marker,
        color="orange",
        alpha=alpha,
        label="Ti XRCS (w)",
    )
    value = raw_data["xrcs"]["te_n3w"].sel(t=slice(tlim[0], tlim[1]))
    error = raw_data["xrcs"]["te_n3w"].error.sel(t=slice(tlim[0], tlim[1]))
    value = xr.where(value > 0, value, np.nan)
    plt.errorbar(
        value.t,
        value * const_temp,
        error * const_temp,
        marker=xrcs_marker,
        color="blue",
        alpha=alpha,
        label="Te XRCS (n3/w)",
    )
    value = raw_data["xrcs"]["te_kw"].sel(t=slice(tlim[0], tlim[1]))
    error = raw_data["xrcs"]["te_kw"].error.sel(t=slice(tlim[0], tlim[1]))
    value = xr.where(value > 0, value, np.nan)
    plt.errorbar(
        value.t,
        value * const_temp,
        error * const_temp,
        marker=xrcs_marker,
        color="cyan",
        alpha=alpha,
        label="Te XRCS (k/w)",
    )
    ylim = plt.ylim()
    plt.xlim(tlim)
    plt.vlines(tplot, ylim[0], ylim[1], color="black", linestyle="dashed")
    plt.ylabel(label_temp)
    plt.xlabel(label_time)
    plt.legend()
    plt.title(f"Pulse {pulse}")
    if savefig:
        save_figure(fig_name=f"{pulse}_time_evol_temperatures", ext=ext)

    plt.figure()
    value = raw_data["efit"]["wp"].sel(t=slice(tlim[0], tlim[1]))
    error = value * 0.2
    value = xr.where(value > 0, value, np.nan)
    plt.errorbar(
        value.t,
        value * const_weq,
        error * const_weq,
        marker=default_marker,
        color="grey",
        alpha=alpha,
        label="Wp EFIT",
    )
    mean = Wastra_all.mean("run")
    if use_std:
        up = mean + Wastra_all.std("run")
        low = mean - Wastra_all.std("run")
    else:
        up = Wastra_all.max("run")
        low = Wastra_all.min("run")
    plt.plot(mean.t, mean * const_weq, color="blue", label="model")
    plt.fill_between(mean.t, up * const_weq, low * const_weq, alpha=alpha, color="blue")
    if plot_all:
        plot_all_runs(runs, Wastra_all * const_weq, alpha, colors)

    mean = Wth_all.mean("run")
    if use_std:
        up = mean + Wth_all.std("run")
        low = mean - Wth_all.std("run")
    else:
        up = Wth_all.max("run")
        low = Wth_all.min("run")
    plt.plot(mean.t, mean * const_weq, color="red", label="ASTRA thermal")
    plt.fill_between(mean.t, up * const_weq, low * const_weq, alpha=alpha, color="red")
    if plot_all:
        plot_all_runs(runs, Wth_all * const_weq, alpha, colors)

    ylim = plt.ylim()
    plt.xlim(tlim)
    plt.vlines(tplot, ylim[0], ylim[1], color="black", linestyle="dashed")
    plt.ylabel(label_wp)
    plt.xlabel(label_time)
    plt.legend()
    plt.title(f"Pulse {pulse}")
    if savefig:
        save_figure(fig_name=f"{pulse}_time_evol_stored_energy", ext=ext)

    plt.figure()
    mean = Ptot_all.mean("run")
    if use_std:
        up = mean + Ptot_all.std("run")
        low = mean - Ptot_all.std("run")
    else:
        up = Ptot_all.max("run")
        low = Ptot_all.min("run")
    plt.plot(mean.t, mean * const_power, color="red", label="$P_{abs}$")
    plt.fill_between(
        mean.t, up * const_power, low * const_power, alpha=alpha, color="red"
    )
    if plot_all:
        plot_all_runs(runs, Ptot_all * const_power, alpha, colors)

    mean = Pabs_all.mean("run")
    if use_std:
        up = mean + Pabs_all.std("run")
        low = mean - Pabs_all.std("run")
    else:
        up = Pabs_all.max("run")
        low = Pabs_all.min("run")
    plt.plot(mean.t, mean * const_power, color="orange", label="$P_{abs}(NBI)$")
    plt.fill_between(
        mean.t, up * const_power, low * const_power, alpha=alpha, color="orange"
    )
    if plot_all:
        plot_all_runs(runs, Pabs_all * const_power, alpha, colors)

    mean = P_oh_all.mean("run")
    if use_std:
        up = mean + P_oh_all.std("run")
        low = mean - P_oh_all.std("run")
    else:
        up = P_oh_all.max("run")
        low = P_oh_all.min("run")
    plt.plot(mean.t, mean * const_power, color="blue", label="$P(OH)$")
    plt.fill_between(
        mean.t, up * const_power, low * const_power, alpha=alpha, color="blue"
    )
    if plot_all:
        plot_all_runs(runs, P_oh_all * const_power, alpha, colors)

    ylim = plt.ylim()
    plt.xlim(tlim)
    plt.vlines(tplot, ylim[0], ylim[1], color="black", linestyle="dashed")
    plt.ylabel(label_power)
    plt.xlabel(label_time)
    plt.legend()
    plt.title(f"Pulse {pulse}")
    if savefig:
        save_figure(fig_name=f"{pulse}_time_heating_power", ext=ext)

    plt.figure()
    Wp_mean = raw_data["efit"]["wp"].sel(t=slice(tlim[0], tlim[1]))
    dWp_dt = Wp_mean.differentiate("t", edge_order=2)
    Ptot_mean = Ptot_all.mean("run").interp(t=Wp_mean.t)
    if use_std:
        Ptot_up = Ptot_mean + Ptot_all.std("run").interp(t=Wp_mean.t)
        Ptot_low = Ptot_mean - Ptot_all.std("run").interp(t=Wp_mean.t)
    else:
        Ptot_up = Ptot_all.max("run").interp(t=Wp_mean.t)
        Ptot_low = Ptot_all.min("run").interp(t=Wp_mean.t)
    taue_mean = Wp_mean / Ptot_mean
    taue_up = Wp_mean / Ptot_low
    taue_low = Wp_mean / Ptot_up
    plt.plot(
        taue_mean.t,
        taue_mean * const_taue,
        color="blue",
        label=r"$\tau_E(EFIT)$ no dW/dt",
    )
    plt.fill_between(
        taue_mean.t,
        taue_up * const_taue,
        taue_low * const_taue,
        alpha=alpha,
        color="blue",
    )
    taue_mean_dw = Wp_mean / (Ptot_mean - dWp_dt.interp(t=Wp_mean.t))
    taue_up_dw = Wp_mean / (Ptot_low - dWp_dt.interp(t=Wp_mean.t))
    taue_low_dw = Wp_mean / (Ptot_up - dWp_dt.interp(t=Wp_mean.t))
    plt.plot(
        taue_mean_dw.t,
        taue_mean_dw * const_taue,
        color="red",
        label=r"$\tau_E(EFIT)$ with dW/dt",
    )
    plt.fill_between(
        taue_mean_dw.t,
        taue_up_dw * const_taue,
        taue_low_dw * const_taue,
        alpha=alpha,
        color="red",
    )

    Wth_mean = Wth_all.mean("run").interp(t=Wp_mean.t)
    dWth_dt = Wth_mean.differentiate("t", edge_order=2)
    taue_th_mean = Wth_mean / Ptot_mean
    taue_th_up = Wth_mean / Ptot_low
    taue_th_low = Wth_mean / Ptot_up
    plt.plot(
        taue_th_mean.t,
        taue_th_mean * const_taue,
        color="cyan",
        label=r"$\tau_E(thermal)$ no dW/dt",
    )
    plt.fill_between(
        taue_th_mean.t,
        taue_th_up * const_taue,
        taue_th_low * const_taue,
        alpha=alpha,
        color="cyan",
    )
    taue_th_mean_dw = Wth_mean / (Ptot_mean - dWth_dt.interp(t=Wth_mean.t))
    taue_th_up_dw = Wth_mean / (Ptot_low - dWth_dt.interp(t=Wth_mean.t))
    taue_th_low_dw = Wth_mean / (Ptot_up - dWth_dt.interp(t=Wth_mean.t))
    plt.plot(
        taue_th_mean_dw.t,
        taue_th_mean_dw * const_taue,
        color="orange",
        label=r"$\tau_E(thermal)$ with dW/dt",
    )
    plt.fill_between(
        taue_th_mean_dw.t,
        taue_th_up_dw * const_taue,
        taue_th_low_dw * const_taue,
        alpha=alpha,
        color="orange",
    )

    ylim = plt.ylim()
    plt.xlim(tlim)
    plt.vlines(tplot, ylim[0], ylim[1], color="black", linestyle="dashed")
    plt.ylabel(label_taue)
    plt.xlabel(label_time)
    plt.legend()
    plt.title(f"Pulse {pulse}")
    if savefig:
        save_figure(fig_name=f"{pulse}_taue", ext=ext)

    plt.figure()
    nTtaue_mean = (
        Ni_all.mean("run").interp(rho_poloidal=0)
        * Ti_all.mean("run").interp(rho_poloidal=0)
        * taue_mean.sel(t=tplot, method="nearest")
    )
    nTtaue_low = (
        Ni_all.min("run").interp(rho_poloidal=0)
        * Ti_all.min("run").interp(rho_poloidal=0)
        * taue_low.sel(t=tplot, method="nearest")
    )
    plt.plot(
        nTtaue_mean.t,
        nTtaue_mean * const_nTtaue,
        color="blue",
        label=r"$n_i(0) T_i(0) \tau_E$(no dWp/dt)",
        marker=default_marker,
    )
    nTtaue_mean_dw = (
        Ni_all.mean("run").interp(rho_poloidal=0)
        * Ti_all.mean("run").interp(rho_poloidal=0)
        * taue_mean_dw.sel(t=tplot, method="nearest")
    )
    nTtaue_low_dw = (
        Ni_all.min("run").interp(rho_poloidal=0)
        * Ti_all.min("run").interp(rho_poloidal=0)
        * taue_low_dw.sel(t=tplot, method="nearest")
    )
    plt.plot(
        nTtaue_mean_dw.t,
        nTtaue_mean_dw * const_nTtaue,
        color="red",
        label=r"$n_i(0) T_i(0) \tau_E$(with dWp/dt)",
        marker=default_marker,
    )

    nTtaue_th_mean = (
        Ni_all.mean("run").interp(rho_poloidal=0)
        * Ti_all.mean("run").interp(rho_poloidal=0)
        * taue_th_mean.sel(t=tplot, method="nearest")
    )
    nTtaue_th_low = (
        Ni_all.min("run").interp(rho_poloidal=0)
        * Ti_all.min("run").interp(rho_poloidal=0)
        * taue_th_low.sel(t=tplot, method="nearest")
    )
    plt.plot(
        nTtaue_th_mean.t,
        nTtaue_th_mean * const_nTtaue,
        color="blue",
        label=r"$n_i(0) T_i(0) \tau_E(thermal)$(no dWp/dt)",
        marker=default_marker,
    )
    nTtaue_th_mean_dw = (
        Ni_all.mean("run").interp(rho_poloidal=0)
        * Ti_all.mean("run").interp(rho_poloidal=0)
        * taue_th_mean_dw.sel(t=tplot, method="nearest")
    )
    nTtaue_th_low_dw = (
        Ni_all.min("run").interp(rho_poloidal=0)
        * Ti_all.min("run").interp(rho_poloidal=0)
        * taue_th_low_dw.sel(t=tplot, method="nearest")
    )
    plt.plot(
        nTtaue_th_mean_dw.t,
        nTtaue_th_mean_dw * const_nTtaue,
        color="red",
        label=r"$n_i(0) T_i(0) \tau_E$(with dWp/dt)",
        marker=default_marker,
    )

    ylim = plt.ylim()
    plt.xlim(tlim)
    plt.vlines(tplot, ylim[0], ylim[1], color="black", linestyle="dashed")
    plt.ylabel(label_nTtaue)
    plt.xlabel(label_time)
    plt.legend()
    plt.title(f"Pulse {pulse}")
    if savefig:
        save_figure(fig_name=f"{pulse}_n_T_taue", ext=ext)

    plt.figure()
    value = raw_data["smmh1"]["ne"].sel(t=slice(tlim[0], tlim[1]))
    value = xr.where(value > 0, value, np.nan)
    plt.plot(
        value.t,
        value * const_dens,
        marker=smmh1_marker,
        color="blue",
        alpha=alpha,
        label="Ne SMM",
    )
    ylim = plt.ylim()
    plt.xlim(tlim)
    plt.vlines(tplot, ylim[0], ylim[1], color="black", linestyle="dashed")
    plt.ylabel(label_dens)
    plt.xlabel(label_time)
    plt.ylim(0,)
    plt.xlim(0.02,)
    plt.legend()
    plt.title(f"Pulse {pulse}")
    if savefig:
        save_figure(fig_name=f"{pulse}_time_evol_density", ext=ext)

    # Profiles
    if multiplot:
        fig, axs = plt.subplots(2, figsize=(7.5, 12))
        ax = axs[0]
    else:
        fig, ax = plt.subplots(1)
    mean = Ne_all.mean("run")
    if use_std:
        up = mean + Ne_all.std("run")
        low = mean - Ne_all.std("run")
    else:
        up = Ne_all.max("run")
        low = Ne_all.min("run")
    ax.plot(mean.rho_poloidal, mean * const_dens, color="blue", label="Electrons")
    ax.fill_between(
        mean.rho_poloidal, up * const_dens, low * const_dens, alpha=alpha, color="blue",
    )
    if plot_all:
        plot_all_runs(runs, Ne_all * const_dens, alpha, colors)

    mean = Ni_all.mean("run")
    if use_std:
        up = mean + Ni_all.std("run")
        low = mean - Ni_all.std("run")
    else:
        up = Ni_all.max("run")
        low = Ni_all.min("run")
    ax.plot(mean.rho_poloidal, mean * const_dens, color="red", label="Thermal ions")
    ax.fill_between(
        mean.rho_poloidal, up * const_dens, low * const_dens, alpha=alpha, color="red",
    )
    if plot_all:
        plot_all_runs(runs, Ni_all * const_dens, alpha, colors)

    mean = Nf_all.mean("run")
    if use_std:
        up = mean + Nf_all.std("run")
        low = mean - Nf_all.std("run")
    else:
        up = Nf_all.max("run")
        low = Nf_all.min("run")
    ax.plot(mean.rho_poloidal, mean * const_dens, color="green", label="Fast ions")
    ax.fill_between(
        mean.rho_poloidal,
        up * const_dens,
        low * const_dens,
        alpha=alpha,
        color="green",
    )
    if plot_all:
        plot_all_runs(runs, Nf_all * const_dens, alpha, colors)
    ax.legend()
    ax.set_xlabel(r"$\rho_{pol}$")
    ax.set_ylabel(label_dens)
    if not multiplot:
        ax.set_title(f"{pulse} Densities @ {int(tplot*1.e3)} ms")
        if savefig:
            save_figure(fig_name=f"{pulse}_el_and_ion_densities_HDA-CXRS", ext=ext)
    # else:
    #     ax.xaxis.set_ticklabels([])

    if multiplot:
        ax = axs[1]
    else:
        fig, ax = plt.subplots(1)
    mean = Te_all.mean("run")
    if use_std:
        up = mean + Te_all.std("run")
        low = mean - Te_all.std("run")
    else:
        up = Te_all.max("run")
        low = Te_all.min("run")
    ax.plot(mean.rho_poloidal, mean * const_temp, color="blue", label="Electrons")
    ax.fill_between(
        mean.rho_poloidal, up * const_temp, low * const_temp, alpha=alpha, color="blue",
    )
    if plot_all:
        plot_all_runs(runs, Te_all * const_temp, alpha, colors)
    ax.errorbar(
        rho_mean_te.mean("run"),
        te_xrcs_bckc.mean("run") * const_temp,
        (te_xrcs_bckc.min("run") - te_xrcs_bckc.max("run")) * const_temp,
        marker=xrcs_marker,
        mfc="white",
        color="blue",
    )
    ax.hlines(
        te_xrcs_bckc.mean("run") * const_temp,
        rho_in_te.min("run"),
        rho_out_te.max("run"),
        color="white",
        linewidth=3,
    )
    ax.hlines(
        te_xrcs_bckc.mean("run") * const_temp,
        rho_in_te.min("run"),
        rho_out_te.max("run"),
        color="blue",
    )

    mean = Ti_all.mean("run")
    if use_std_ti:
        up = mean + Ti_all.std("run")
        low = mean - Ti_all.std("run")
    else:
        up = Ti_all.max("run")
        low = Ti_all.min("run")
    print(f"Ti(0)   = {mean.sel(rho_poloidal=0).values}")
    print(f"  error = {(up - mean).sel(rho_poloidal=0).values}")
    ax.plot(mean.rho_poloidal, mean * const_temp, color="red", label="Ions")
    ax.fill_between(
        mean.rho_poloidal, up * const_temp, low * const_temp, alpha=alpha, color="red",
    )
    if plot_all:
        plot_all_runs(runs, Ti_all * const_temp, alpha, colors)
    ax.errorbar(
        rho_mean_ti.mean("run"),
        ti_xrcs_bckc.mean("run") * const_temp,
        (ti_xrcs_bckc.min("run") - ti_xrcs_bckc.max("run")) * const_temp,
        marker=xrcs_marker,
        color="red",
        mfc="white",
        label="XRCS",
    )
    ax.hlines(
        ti_xrcs_bckc.mean("run") * const_temp,
        rho_in_ti.min("run"),
        rho_out_ti.max("run"),
        color="white",
        linewidth=3,
    )
    ax.hlines(
        ti_xrcs_bckc.mean("run") * const_temp,
        rho_in_ti.min("run"),
        rho_out_ti.max("run"),
        color="red",
    )
    quantity = "ti"
    for analysis_key in cxrs_analyses.keys():
        key = f"{quantity}_{analysis_key}"
        ti = (xr.where(cxrs[key] > 0, cxrs[key], np.nan) * const_temp).sel(
            t=tplot, method="nearest"
        )
        ti_error = (
            xr.where(cxrs[key] > 0, cxrs[key].attrs["error"], np.nan) * const_temp
        ).sel(t=tplot, method="nearest")
        if np.any(ti > 0):
            ax.errorbar(
                cxrs[key].rho_poloidal,
                ti,
                ti_error,
                marker=cxrs_markers[analysis_key],
                mfc="white",
                color="red",
                label="CXRS",#cxrs[key].name,
                linestyle="",
            )
    if plot_bckc:
        for ti in bckc_ti:
            ax.scatter(bckc_rho, ti * const_temp, marker="x", color="black")
    ax.legend()
    ax.set_xlabel(r"$\rho_{pol}$")
    ax.set_ylabel("(keV)")
    if not multiplot:
        if savefig:
            plt.title(f"{pulse} temperatures @ {int(tplot*1.e3)} ms")
            save_figure(fig_name=f"{pulse}_temperatures_HDA-CXRS", ext=ext)
    else:
        if savefig:
            save_figure(fig_name=f"{pulse}_temperatures_and_densities_HDA-CXRS", ext=ext)

    const_rot = 1.0e-3
    plt.figure()
    mean = Ti_all.mean("run") / Ti_all.mean("run").sel(rho_poloidal=0) * omega_scaling
    if use_std_vtor:
        up = (
            (Ti_all.mean("run") + Ti_all.std("run"))
            / Ti_all.mean("run").sel(rho_poloidal=0)
            * omega_scaling
        )
        low = (
            (Ti_all.mean("run") - Ti_all.std("run"))
            / Ti_all.mean("run").sel(rho_poloidal=0)
            * omega_scaling
        )
    else:
        up = Ti_all.max("run") / Ti_all.mean("run").sel(rho_poloidal=0) * omega_scaling
        low = Ti_all.min("run") / Ti_all.mean("run").sel(rho_poloidal=0) * omega_scaling
    plt.plot(mean.rho_poloidal, mean * const_rot, color="red", label="Ions")
    plt.fill_between(
        mean.rho_poloidal, up * const_rot, low * const_rot, alpha=alpha, color="red"
    )
    quantity = "vtor"
    for analysis_key in cxrs_analyses.keys():
        key = f"{quantity}_{analysis_key}"
        vtor = (xr.where(cxrs[key] > 0, cxrs[key], np.nan)).sel(
            t=tplot, method="nearest"
        )
        vtor_error = (xr.where(cxrs[key] > 0, cxrs[key].attrs["error"], np.nan)).sel(
            t=tplot, method="nearest"
        )
        omega = vtor / vtor.R  # 0.5 #
        omega_error = vtor_error / omega.R

        if np.any(omega > 0):
            plt.errorbar(
                cxrs[key].rho_poloidal,
                omega * const_rot,
                omega_error * const_rot,
                marker=cxrs_markers[analysis_key],
                mfc="white",
                color="red",
                label="CXRS", #cxrs[key].name,
                linestyle="",
            )
    if plot_bckc:
        for vtor in bckc_vtor:
            plt.scatter(bckc_rho, vtor * const_temp, marker="x", color="black")
    plt.title(f"{pulse} Toroidal rotation @ {int(tplot*1.e3)} ms")
    plt.legend(fontsize=10)
    plt.xlabel(r"$\rho_{pol}$")
    plt.ylabel("(krad/s)")
    if savefig:
        save_figure(fig_name=f"{pulse}_toroidal_rotation_HDA-CXRS", ext=ext)
    plt.figure()
    mean = NAr_all.mean("run")
    if use_std:
        up = mean + NAr_all.std("run")
        low = mean - NAr_all.std("run")
    else:
        up = NAr_all.max("run")
        low = NAr_all.min("run")
    plt.plot(mean.rho_poloidal, mean * const_imp, color="orange", label="Argon")
    plt.fill_between(
        mean.rho_poloidal, up * const_imp, low * const_imp, alpha=alpha, color="orange",
    )
    if plot_all:
        plot_all_runs(runs, NAr_all * const_imp, alpha, colors)
    plt.title(f"{pulse} Impurity density @ {int(tplot*1.e3)} ms")
    plt.legend()
    plt.xlabel(r"$\rho_{pol}$")
    plt.ylabel("($10^{16}$ $m^{-3}$)")
    if savefig:
        save_figure(fig_name=f"{pulse}_argon_density_HDA-CXRS", ext=ext)

    print("\n")
    print(
        f"Ptot_abs (MW) = {Ptot_all.mean('run').sel(t=tplot, method='nearest').values * 1.e-6:.2f}"
    )
    print(
        f"Poh (MW) = {P_oh_all.mean('run').sel(t=tplot, method='nearest').values * 1.e-6:.2f}"
    )
    print(
        f"Pnbi_inj (MW) = {Pnb_all.mean('run').sel(t=tplot, method='nearest').values * 1.e-6:.2f}"
    )
    print(
        f"Pnbi_abs (MW) = {Pabs_all.mean('run').sel(t=tplot, method='nearest').values * 1.e-6:.2f}"
    )
    print(f"Wp EFIT (kJ) = {Wp_mean.sel(t=tplot, method='nearest').values * 1.e-3:.2f}")
    print(
        f"Wthermal ASTRA (kJ) = {Wth_mean.sel(t=tplot, method='nearest').values * 1.e-3:.2f}"
    )
    print(
        f"Ne(0) (1e19 m^-3) = {Ne_all.mean('run').sel(rho_poloidal=0).values * 1.e-19:.2f}"
    )
    print(
        f"Ni(0) (1e19 m^-3) = {Ni_all.mean('run').sel(rho_poloidal=0).values * 1.e-19:.2f}"
    )
    print(
        f"Nf(0) (1e19 m^-3) = {Nf_all.mean('run').sel(rho_poloidal=0).values * 1.e-19:.2f}"
    )
    print(f"Te(0) (keV) = {Te_all.mean('run').sel(rho_poloidal=0).values * 1.e-3:.2f}")
    print(f"Ti(0) (keV) = {Ti_all.mean('run').sel(rho_poloidal=0).values * 1.e-3:.2f}")
    taue = taue_mean.sel(t=tplot, method="nearest").values * 1.0e3
    taue_dw = taue_mean_dw.sel(t=tplot, method="nearest").values * 1.0e3
    print(f"Tau_E(EFIT) with/without dWp/dt (ms) = {taue_dw:.2f} / {taue:.2f}")
    taue = taue_th_mean.sel(t=tplot, method="nearest").values * 1.0e3
    taue_dw = taue_th_mean_dw.sel(t=tplot, method="nearest").values * 1.0e3
    print(f"Tau_E(thermal) with/without dWp/dt (ms) = {taue_dw:.2f} / {taue:.2f}")
    nttau = nTtaue_mean.values * 1.0e-3 * 1.0e-18
    nttau_dw = nTtaue_mean_dw.values * 1.0e-3 * 1.0e-18
    print(
        f"EFIT Ni(0) Ti(0) Tau_E with/without dW/dt (1.e18 m^-3 keV s) = {nttau_dw:.2f} / {nttau:.2f}"
    )
    nttau = nTtaue_low.values * 1.0e-3 * 1.0e-18
    nttau_dw = nTtaue_low_dw.values * 1.0e-3 * 1.0e-18
    print(f"...low limits = {nttau_dw:.2f} / {nttau:.2f}")
    nttau = nTtaue_th_mean.values * 1.0e-3 * 1.0e-18
    nttau_dw = nTtaue_th_mean_dw.values * 1.0e-3 * 1.0e-18
    print(
        f"Thermal Ni(0) Ti(0) Tau_E with/without dW/dt (1.e18 m^-3 keV s) = {nttau_dw:.2f} / {nttau:.2f}"
    )
    nttau = nTtaue_th_low.values * 1.0e-3 * 1.0e-18
    nttau_dw = nTtaue_th_low_dw.values * 1.0e-3 * 1.0e-18
    print(f"...low limits = {nttau_dw:.2f} / {nttau:.2f}")


def plot_all_runs(runs, values, alpha, colors, label=True):
    for i, run in enumerate(runs):
        values.sel(run=run).plot(
            alpha=alpha, linestyle="dashed", color=colors[i], label=run,
        )
