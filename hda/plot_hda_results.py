import matplotlib.pylab as plt
import numpy as np
from matplotlib import rcParams, cm
from xarray import DataArray
import xarray as xr
from copy import deepcopy
from scipy import constants

from indica.readers import ST40Reader
from hda.read_st40 import ST40data
from indica.equilibrium import Equilibrium
from indica.converters.time import bin_in_time_dt

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


def set_sizes_profiles(fontsize=15, legendsize=13, markersize=9):
    rcParams.update({"font.size": fontsize})
    rcParams.update({"legend.fontsize": legendsize})
    rcParams.update({"lines.markersize": markersize})
    rcParams.update({"lines.width": markersize})


def set_sizes_time_evol(fontsize=10, weight=600, legendsize=10, markersize=5):
    rcParams.update({"font.size": fontsize})
    rcParams.update({"font.weight": weight})
    rcParams.update({"legend.fontsize": legendsize})
    rcParams.update({"lines.markersize": markersize})
    rcParams.update({"lines.linewidth": 2})


def available_quantities():

    qdict = {
        "efit:ipla": {
            "const": 1.0e-6,
            "label": "$I_P$ EFIT",
            "ylabel": "(MA)",
            "ylim": (0, None),
        },
        "efit:wp": {
            "const": 1.0e-3,
            "label": "$W_P$ EFIT",
            "ylabel": "(kJ)",
            "ylim": (0, None),
        },
        "nbi:pin": {
            "const": 1.0e-6,
            "label": "$P_{NBI}$",
            "ylabel": "(MW)",
            "ylim": (0, None),
            "add_pulse_label":True,
        },
        "mag:vloop": {"const": 1.0, "ylabel": "$V_{loop}$ (V)", "ylim": (0, None),},
        "smmh1:ne_bar": {
            "const": 1.0e-19,
            "label": "$\overline{N}_e$ SMM",
            "ylabel": "($10^{19}$ $m^{-3}$)",
            "ylim": (0, 7),
        },
        "lines:h_alpha": {
            "const": 1.0,
            "label": r"$H_\alpha$ Filter",
            "ylabel": r"(a.u.)",
            "ylim": (0, 0.5),
        },
        "diode_detr:filter_001": {
            "const": 1.0e-3,
            "label": "$P_{SXR}$",
            "ylabel": "(a.u.)",
            "ylim": (0, None),
        },
        "xrcs:te_avrg": {
            "const": 1.0e-3,
            "label": "$T_e$ XRCS",
            "ylabel": "(keV)",
            "ylim": (0, None),
            "error": True,
            "marker": "o",
        },
        "xrcs:ti_w": {
            "const": 1.0e-3,
            "label": "$T_i$ XRCS",
            "ylabel": "(keV)",
            "ylim": (0, 10),
            "error": True,
            "marker": "o",
        },
        "cxrs:ti_ff": {
            "const": 1.0e-3,
            "label": "$T_i$ CXRS",
            "ylabel": "(keV)",
            "ylim": (0, 10),
            "error": True,
            "marker": "x",
        },
        "cxrs:vtor_ff": {
            "const": 1.0e-3,
            "label": "$V_{tor}$ CXRS",
            "ylabel": "(km/s)",
            "ylim": (0, 300),
            "error": True,
            "marker": "x",
        },
        "mhd:ampl_odd_n": {
            "const": 1.0,
            "label": "Odd n MHD",
            "ylabel": "(a.u.)",
            "ylim": (0, None),
        },
    }

    return qdict


def save_figure(fig_name="", orientation="landscape", ext="png"):
    _file = f"/home/marco.sertoli/figures/Indica/{fig_name}.{ext}"
    plt.savefig(
        _file, orientation=orientation, dpi=300, pil_kwargs={"quality": 95},
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

    # # Latest after discovery of flipped optics 23/09/2022
    # R_corrected = np.array([0.8794, 0.6345, 0.5173, 0.4583, 0.4206]) + R_shift
    # angle_correction = [1.0, 1.0, 1.0, 1.0, 1.0]

    # RUN numbers: ff=full-fit, bs=background-subtraction (status @ 04/07/2022)
    pulse_info = {
        10014: {"ff": 7, "bs": 9},
        10009: {"ff": 3, "bs": 1},
        9831: {"ff": 5, "bs": 5},
        9787: {"ff": 3},
        9783: {"ff": 2},
        9780: {"ff": 5, "bs": 6},
        9539: {"ff": 2},
        9520: {"ff": 2},
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
    def calc_mean_std(dataarray: DataArray, runs: list, use_std=False):
        _data = []
        for run in runs:
            _data.append(dataarray.sel(run=run))

        data = xr.concat(_data, "run").assign_coords({"run": runs})

        mean = data.mean("run")
        std = data.std("run")
        if use_std:
            up = mean + std
            low = mean - std
        else:
            up = data.max("run")
            low = data.min("run")

        return mean, std, up, low

    set_sizes()

    all_runs = [str(run) for run in np.arange(60, 76 + 1)]
    R_shift = 0.0
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
        keep = ["60", "64", "65", "66", "72", "73"]
        run_plus_astra = 500
        run_add_hda = "MID"
        omega_scaling = 440e3
        R_shift = 0.03
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
    elif pulse == 9520:
        tplot = 0.087
        keep = ["61", "62", "69", "70"]
        run_plus_astra = 500
        run_add_hda = "MID"
        omega_scaling = 550e3
        R_shift = 0.0
    elif pulse == 9539:
        tplot = 0.072
        keep = ["62", "64", "69", "70", "72"]
        run_plus_astra = 500
        run_add_hda = "MID"
        omega_scaling = 550e3
        R_shift = 0.03
    elif pulse == 9787:
        # raise ValueError("HDA systematically overestimates stored energy for pulse 9787!!!!")
        tplot = 0.098
        keep = ["60", "66", "68", "73", "74", "76"]
        run_plus_astra = 500
        run_add_hda = "MID"
        omega_scaling = 550e3
        R_shift = 0.03
    else:
        raise ValueError(f"...input missing for pulse {pulse}..")

    cmap = cm.rainbow
    varr = np.linspace(0, 1, len(all_runs))
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
    # for run in keep:
    for run in pl_dict_all.keys():
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
    for run in runs:
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

        if "te_n3w" in bckc_dict[run]["xrcs"].keys():
            te_key = "te_n3w"
        else:
            te_key = "te_kw"
        te_tmp = bckc_dict[run]["xrcs"][te_key].sel(t=tplot, method="nearest")
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
    for analysis_key, analysis in cxrs_analyses.items():
        for quantity in ["ti", "vtor"]:
            key = f"{quantity}_{analysis_key}"
            if key not in cxrs.keys():
                continue
            cxrs[key].name = analysis
            cxrs[key] = cxrs[key].assign_coords(z=("R", xr.full_like(cxrs[key].R, 0)))

            cxrs_rho = equilibrium.rho.interp(t=tplot, method="nearest").drop_vars("t")
            cxrs_rho = cxrs_rho.interp(R=cxrs[key].R, z=cxrs[key].z)
            cxrs[key] = cxrs[key].assign_coords(rho_poloidal=("R", cxrs_rho))

    # Time evolution
    plt.figure()
    if "ti_ff" in raw_data["cxrs"].keys():
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

    if "ti_bs" in raw_data["cxrs"].keys():
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

    if "te_n3w" in bckc_dict[run]["xrcs"].keys():
        te_key = "te_n3w"
    else:
        te_key = "te_kw"
    value = raw_data["xrcs"][te_key].sel(t=slice(tlim[0], tlim[1]))
    error = raw_data["xrcs"][te_key].error.sel(t=slice(tlim[0], tlim[1]))
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

    mean, std, up, low = calc_mean_std(Wastra_all, keep)
    plt.plot(mean.t, mean * const_weq, color="blue", label="model")
    plt.fill_between(mean.t, up * const_weq, low * const_weq, alpha=alpha, color="blue")
    if plot_all:
        plot_all_runs(runs, Wastra_all * const_weq, alpha, colors)

    mean, std, up, low = calc_mean_std(Wth_all, keep)
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
    mean, std, up, low = calc_mean_std(Ptot_all, keep)
    plt.plot(mean.t, mean * const_power, color="red", label="$P_{abs}$")
    plt.fill_between(
        mean.t, up * const_power, low * const_power, alpha=alpha, color="red"
    )
    if plot_all:
        plot_all_runs(runs, Ptot_all * const_power, alpha, colors)

    mean, std, up, low = calc_mean_std(Pabs_all, keep)
    plt.plot(mean.t, mean * const_power, color="orange", label="$P_{abs}(NBI)$")
    plt.fill_between(
        mean.t, up * const_power, low * const_power, alpha=alpha, color="orange"
    )
    if plot_all:
        plot_all_runs(runs, Pabs_all * const_power, alpha, colors)

    mean, std, up, low = calc_mean_std(P_oh_all, keep)
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
    profile_data = {}
    if multiplot:
        fig, axs = plt.subplots(2, figsize=(7.5, 12))
        ax = axs[0]
    else:
        fig, ax = plt.subplots(1)

    mean, std, up, low = calc_mean_std(Ne_all, keep)
    ax.plot(mean.rho_poloidal, mean * const_dens, color="blue", label="Electrons")
    ax.fill_between(
        mean.rho_poloidal, up * const_dens, low * const_dens, alpha=alpha, color="blue",
    )
    if plot_all:
        plot_all_runs(runs, Ne_all * const_dens, alpha, colors)
    profile_data["Ne"] = {"mean": mean, "err": std}

    mean, std, up, low = calc_mean_std(Ni_all, keep)
    ax.plot(mean.rho_poloidal, mean * const_dens, color="red", label="Thermal ions")
    ax.fill_between(
        mean.rho_poloidal, up * const_dens, low * const_dens, alpha=alpha, color="red",
    )
    if plot_all:
        plot_all_runs(runs, Ni_all * const_dens, alpha, colors)
    profile_data["Ni"] = {"mean": mean, "err": std}

    mean = Nf_all.mean("run")
    std = Nf_all.std("run")
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
    profile_data["Nf"] = {"mean": mean, "err": std}
    ax.legend()
    ax.set_xlabel(r"$\rho_{pol}$")
    ax.set_ylabel(label_dens)
    if not multiplot:
        ax.set_title(f"{pulse} Densities @ {int(tplot*1.e3)} ms")
        if savefig:
            save_figure(fig_name=f"{pulse}_el_and_ion_densities_HDA-CXRS", ext=ext)

    if multiplot:
        ax = axs[1]
    else:
        fig, ax = plt.subplots(1)

    mean, std, up, low = calc_mean_std(Te_all, keep)
    ax.plot(mean.rho_poloidal, mean * const_temp, color="blue", label="Electrons")
    ax.fill_between(
        mean.rho_poloidal, up * const_temp, low * const_temp, alpha=alpha, color="blue",
    )
    if plot_all:
        plot_all_runs(runs, Te_all * const_temp, alpha, colors)
    profile_data["Te"] = {"mean": mean, "err": std}

    rho_xrcs_mean, _, _, _ = calc_mean_std(rho_mean_te, keep)
    rho_xrcs_in, _, _, _ = calc_mean_std(rho_in_te, keep)
    rho_xrcs_out, _, _, _ = calc_mean_std(rho_out_te, keep)
    te_xrcs_mean, _, te_xrcs_up, te_xrcs_low = calc_mean_std(te_xrcs_bckc, keep)
    ax.errorbar(
        rho_xrcs_mean,
        te_xrcs_mean * const_temp,
        (te_xrcs_up - te_xrcs_low) * const_temp,
        marker=xrcs_marker,
        mfc="white",
        color="blue",
    )
    ax.hlines(
        te_xrcs_mean * const_temp,
        rho_xrcs_in,
        rho_xrcs_out,
        color="white",
        linewidth=3,
    )
    ax.hlines(
        te_xrcs_mean * const_temp, rho_xrcs_in, rho_xrcs_out, color="blue",
    )
    profile_data["Te_xrcs"] = {
        "mean": te_xrcs_mean,
        "err": (te_xrcs_up - te_xrcs_low),
        "rho_mean": rho_xrcs_mean,
        "rho_in": rho_xrcs_in,
        "rho_out": rho_xrcs_out,
    }

    mean, std, up, low = calc_mean_std(Ti_all, keep, use_std=use_std_ti)
    print(f"Ti(0)   = {mean.sel(rho_poloidal=0).values}")
    print(f"  error = {(up - mean).sel(rho_poloidal=0).values}")
    ax.plot(mean.rho_poloidal, mean * const_temp, color="red", label="Ions")
    ax.fill_between(
        mean.rho_poloidal, up * const_temp, low * const_temp, alpha=alpha, color="red",
    )
    if plot_all:
        plot_all_runs(runs, Ti_all * const_temp, alpha, colors)
    profile_data["Ti"] = {"mean": mean, "err": std}

    rho_xrcs_mean, _, _, _ = calc_mean_std(rho_mean_ti, keep)
    rho_xrcs_in, _, _, _ = calc_mean_std(rho_in_ti, keep)
    rho_xrcs_out, _, _, _ = calc_mean_std(rho_out_ti, keep)
    ti_xrcs_mean, _, ti_xrcs_up, ti_xrcs_low = calc_mean_std(ti_xrcs_bckc, keep)
    ax.errorbar(
        rho_xrcs_mean,
        ti_xrcs_mean * const_temp,
        (ti_xrcs_up - ti_xrcs_low) * const_temp,
        marker=xrcs_marker,
        mfc="white",
        color="red",
    )
    ax.hlines(
        ti_xrcs_mean * const_temp,
        rho_xrcs_in,
        rho_xrcs_out,
        color="white",
        linewidth=3,
    )
    ax.hlines(
        ti_xrcs_mean * const_temp, rho_xrcs_in, rho_xrcs_out, color="red",
    )
    profile_data["Ti_xrcs"] = {
        "mean": ti_xrcs_mean,
        "err": (ti_xrcs_up - ti_xrcs_low),
        "rho_mean": rho_xrcs_mean,
        "rho_in": rho_xrcs_in,
        "rho_out": rho_xrcs_out,
    }

    quantity = "ti"
    ti_cxrs = []
    ti_err_cxrs = []
    ti_rho_cxrs = []
    for analysis_key in cxrs_analyses.keys():
        key = f"{quantity}_{analysis_key}"
        if key not in cxrs.keys():
            continue
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
                label="CXRS",  # cxrs[key].name,
                linestyle="",
            )
            ti_cxrs.append(ti)
            ti_err_cxrs.append(ti_error)
            ti_rho_cxrs.append(cxrs[key].rho_poloidal)
    if len(ti_cxrs) > 0:
        profile_data["Ti_cxrs"] = {
            "mean": np.array(ti_cxrs),
            "err": np.array(ti_err_cxrs),
            "rho_mean": ti_rho_cxrs,
        }
    ax.legend()
    ax.set_xlabel(r"$\rho_{pol}$")
    ax.set_ylabel("(keV)")
    if not multiplot:
        if savefig:
            plt.title(f"{pulse} temperatures @ {int(tplot*1.e3)} ms")
            save_figure(fig_name=f"{pulse}_temperatures_HDA-CXRS", ext=ext)
    else:
        if savefig:
            save_figure(
                fig_name=f"{pulse}_temperatures_and_densities_HDA-CXRS", ext=ext
            )

    const_rot = 1.0e-3
    plt.figure()

    _mean, _std, _up, _low = calc_mean_std(Ti_all, keep)
    mean = _mean / _mean.sel(rho_poloidal=0) * omega_scaling
    if use_std_vtor:
        up = (_mean + _std) / _mean.sel(rho_poloidal=0) * omega_scaling
        low = (_mean - _std) / _mean.sel(rho_poloidal=0) * omega_scaling
    else:
        up = _up / _mean.sel(rho_poloidal=0) * omega_scaling
        low = _low / _mean.sel(rho_poloidal=0) * omega_scaling
    plt.plot(mean.rho_poloidal, mean * const_rot, color="red", label="Ions")
    plt.fill_between(
        mean.rho_poloidal, up * const_rot, low * const_rot, alpha=alpha, color="red"
    )
    quantity = "vtor"
    for analysis_key in cxrs_analyses.keys():
        if key not in cxrs.keys():
            continue
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
                label="CXRS",  # cxrs[key].name,
                linestyle="",
            )
    plt.title(f"{pulse} Toroidal rotation @ {int(tplot*1.e3)} ms")
    plt.legend(fontsize=10)
    plt.xlabel(r"$\rho_{pol}$")
    plt.ylabel("(krad/s)")
    if savefig:
        save_figure(fig_name=f"{pulse}_toroidal_rotation_HDA-CXRS", ext=ext)

    plt.figure()

    mean, std, up, low = calc_mean_std(NAr_all, keep)
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

    plt.figure()
    Wp_mean = raw_data["efit"]["wp"].sel(t=slice(tlim[0], tlim[1]))
    dWp_dt = Wp_mean.differentiate("t", edge_order=2)

    Ptot_mean, _, Ptot_up, Ptot_low = calc_mean_std(Ptot_all.interp(t=Wp_mean.t), keep)
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

    Wth_mean, _, Wth_up, Wth_low = calc_mean_std(Wth_all.interp(t=Wp_mean.t), keep)
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
    Ni_mean, _, Ni_up, Ni_low = calc_mean_std(Ni_all.interp(rho_poloidal=0), keep)
    Ti_mean, _, Ti_up, Ti_low = calc_mean_std(Ti_all.interp(rho_poloidal=0), keep)
    nTtaue_mean = Ni_mean * Ti_mean * taue_mean
    nTtaue_low = Ni_low * Ti_low * taue_low
    nTtaue_mean_dw = Ni_mean * Ti_mean * taue_mean_dw
    plt.plot(
        nTtaue_mean.t,
        nTtaue_mean * const_nTtaue,
        color="blue",
        label=r"$n_i(0) T_i(0) \tau_E$(no dWp/dt)",
        marker=default_marker,
    )
    nTtaue_low_dw = Ni_low * Ti_low * taue_low_dw
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
        * taue_th_mean
    )
    nTtaue_th_low = Ni_low * Ti_low * taue_th_low
    plt.plot(
        nTtaue_th_mean.t,
        nTtaue_th_mean * const_nTtaue,
        color="blue",
        label=r"$n_i(0) T_i(0) \tau_E(thermal)$(no dWp/dt)",
        marker=default_marker,
    )
    nTtaue_th_mean_dw = Ni_mean * Ti_mean * taue_th_mean_dw
    nTtaue_th_low_dw = Ni_low * Ti_low * taue_th_low_dw
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

    import pickle

    pickle.dump(
        profile_data,
        open(f"/home/marco.sertoli/data/Indica/{pulse}_profiles_for_Jon.pkl", "wb"),
    )

    return profile_data


def plot_all_runs(runs, values, alpha, colors, label=True):
    for i, run in enumerate(runs):
        values.sel(run=run).plot(
            alpha=alpha, linestyle="dashed", color=colors[i], label=run,
        )


def compare_pulses_prl(data=None):

    data = compare_pulses(data=data, qpop=["mhd:ampl_odd_n", "lines:h_alpha"])

    return data


def compare_pulses_aps(data=None, savefig=False):

    # 9520, 9538, 9783, 9831

    data = compare_pulses(
        [9520, 9539, 9783, 10009],
        data=data,
        qpop=[
            "mhd:ampl_odd_n",
            "mag:vloop",
            "diode_detr:filter_001",
            "xrcs:te_avrg",
            "xrcs:ti_w",
            "cxrs:ti_ff",
            "cxrs:vtor_ff",
        ],
        figname="kinetics",
        savefig=savefig,
    )

    data = compare_pulses(
        [9520, 9539, 9783, 10009],
        data=data,
        qpop=[
            "mhd:ampl_odd_n",
            "mag:vloop",
            "efit:ipla",
            "efit:wp",
            "smmh1:ne_bar",
            "lines:h_alpha",
            "nbi:pin",
        ],
        savefig=savefig,
    )

    return data


def compare_pulses(  # 9783, 9781, 9831, 10013,
    pulses: list = [9538, 9780, 9783, 9831, 10014],
    tstart: float = 0,
    tend: float = 0.15,
    dt: float = 0.003,
    alpha: float = 0.9,
    xlabel: str = "Time (s)",
    figname="",
    savefig: bool = False,
    qpop: list = [""],
    data: list = None,
    R_cxrs: float = 0.4649,
):
    """
    Compare time traces of different pulses
    for APS:
    compare_pulses(qpop=["lines:h_alpha", "mhd:ampl_odd_n"])
    """

    def plot_quantity(qkey: str, linestyle="solid", ):
        if "label" not in qdict[qkey]:
            qdict[qkey]["label"] = ""
        if "marker" not in qdict[qkey].keys():
            qdict[qkey]["marker"] = ""

        diag, quant = qkey.split(":")
        label = ""
        binned = np.array([])
        add_pulse_label = False
        if "add_pulse_label" in qdict[qkey].keys():
            add_pulse_label = qdict[qkey]["add_pulse_label"]
        for i in range(len(data)):
            if add_pulse_label:
                label = str(pulses[i])
            val = data[i][diag][quant] * qdict[qkey]["const"]
            try:
                _binned = bin_in_time_dt(val.t.min() + dt, val.t.max() - dt, dt, val)
                binned = np.append(binned, _binned.values).flatten()
            except ValueError:
                _ = np.nan

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
    for q in qpop:
        if q in qdict.keys():
            qdict.pop(q)

    iax = -1
    set_sizes_time_evol()

    ncols = 4
    npulses = len(pulses)
    if npulses > ncols:
        ncols = npulses
    cmap = cm.gnuplot2
    varr = np.linspace(0.1, 0.75, ncols, dtype=float)
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

            _data = raw_data["cxrs"]["ti_ff"].sel(R=R_cxrs, method="nearest")
            _data = xr.where(_data > 0, _data, np.nan)
            _error = raw_data["cxrs"]["ti_ff"].error.sel(R=R_cxrs, method="nearest")
            _error = xr.where(_error > 0, _error, np.nan)
            raw_data["cxrs"]["ti_ff"] = _data
            raw_data["cxrs"]["ti_ff"].attrs["error"] = _error

            _data = raw_data["cxrs"]["vtor_ff"].sel(R=R_cxrs, method="nearest")
            _data = xr.where(_data > 0, _data, np.nan)
            _error = raw_data["cxrs"]["vtor_ff"].error.sel(R=R_cxrs, method="nearest")
            _error = xr.where(_error > 0, _error, np.nan)
            raw_data["cxrs"]["vtor_ff"] = _data
            raw_data["cxrs"]["vtor_ff"].attrs["error"] = _error

            raw_data["nbi"] = {"pin": raw_data["hnbi1"]["pin"] + raw_data["rfx"]["pin"]}

            smmh1_l = 2 * (raw_data["efit"]["rmjo"] - raw_data["efit"]["rmji"]).sel(
                rho_poloidal=1
            )
            raw_data["smmh1"]["ne_bar"] = raw_data["smmh1"]["ne"] / smmh1_l.interp(
                t=raw_data["smmh1"]["ne"].t
            )

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

    if savefig:
        name = ""
        for pulse in pulses:
            name += f"_{pulse}"
        if len(figname) >1:
            name += f"_{figname}"
        save_figure(fig_name=f"time_evolution_comparison{name}")

    return data


def data_time_evol(
    pulse, tstart=-0.01, tend=0.2, savefig=False, name="", title="", plot_100M=False,
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
        bottom=np.floor(binned.min().values), top=np.ceil(binned.max()),
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
        tmp.t, tmp.values, err.values, label="T$_e$(Ar)", marker="o", alpha=0.9,
    )

    tmp = raw_data["xrcs"]["ti_w"] * const
    err = raw_data["xrcs"]["ti_w"].error * const
    err = xr.where(err > 0, err, np.nan)
    ax.errorbar(
        tmp.t, tmp.values, err.values, label="T$_i$(Ar)", marker="o", alpha=0.9,
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

    return st40_data


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
            color=col, label=run, linestyle="dashed", linewidth=3,
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
                    color=col, label=run, linestyle="dashed", linewidth=3,
                )
                if best_hda in run and hasattr(pl, "el_temp_hi"):
                    Te_err = (
                        pl.el_temp_hi.sel(t=t, method="nearest")
                        - pl.el_temp_lo.sel(t=t, method="nearest")
                    ) / 2.0
                    plt.fill_between(
                        pl.rho, Te - Te_err, Te + Te_err, color=col, alpha=0.8,
                    )
            if k == "Ti":
                Ti = pl.ion_temp.sel(element="ar").sel(t=t, method="nearest")
                Ti.plot(
                    color=col, label=run, linestyle="dashed", linewidth=3,
                )
                if best_hda in run and hasattr(pl, "el_temp_hi"):
                    Ti_err = (
                        pl.ion_temp_hi.sel(element="ar").sel(t=t, method="nearest")
                        - pl.ion_temp_lo.sel(element="ar").sel(t=t, method="nearest")
                    ) / 2.0
                    plt.fill_between(
                        pl.rho, Ti - Ti_err, Ti + Ti_err, color=col, alpha=0.8,
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
