"""Various miscellaneous functions for handling of atomic data."""

import getpass
import numpy as np
from indica.readers import ST40Reader
import matplotlib.pylab as plt
from copy import deepcopy

# First pulse after Boronisation
BORONISATION = [8441]

plt.ion()


def run_default(pulse_start, pulse_end, t=0.03, dt=0.01, savefig=False):
    plot_trends(read_data(pulse_start, pulse_end, t=t, dt=dt), savefig=savefig)


def read_data(
    pulse_start, pulse_end, t=0.03, dt=0.015,
):
    """
    Read data from AVANTES spectrometer and analyse trends
    in line intensity evolution

    Parameters
    ----------
    pulse_start
        first pulse in range to be analysed
    pulse_end
        last pulse in range to be analysed
    t
        centre of time window to use for averaging
    dt
        half-width of time-window to use for averaging

    Returns
    -------

    """

    if type(t) is not list:
        t = [t]
    t = np.array(t)

    tstart = t - dt
    tend = t + dt
    pulses = np.arange(pulse_start, pulse_end + 1)

    if len(t)>1:
        lines = [
            "h_i_656",
            "he_ii_469",
            "b_v_494",
            "o_iv_306",
            "ar_ii_443",
        ]
    else:
        lines = [
            "h_i_656",
            "he_i_588",
            "he_ii_469",
            "b_ii_207",
            "b_iv_449",
            "b_v_494",
            "o_iii_305",
            "o_iii_327",
            "o_iv_306",
            "o_v_278",
            "ar_ii_488",
            "ar_ii_481",
            "ar_ii_474",
            "ar_ii_443",
            "ar_iii_331",
            "ar_iii_330",
            "ar_iii_329",
        ]

    params = {
        "t": t,
        "dt": dt,
        "pulse_start": pulse_start,
        "pulse_end": pulse_end,
        "tstart": tstart,
        "tend": tend,
        "pulses": [],
        "lines_labels": lines,
    }
    results = {
        "puff": [],
        "ipla": {"avrg": [], "stdev": []},
        "wp": {"avrg": [], "stdev": []},
        "nirh1": {"avrg": [], "stdev": []},
        "brems_pi": {"avrg": [], "stdev": []},
        "brems_mp": {"avrg": [], "stdev": []},
        "xrcs_te": {"avrg": [], "stdev": []},
        "xrcs_ti": {"avrg": [], "stdev": []},
        "nbi": {"avrg": [], "stdev": []},
        "lines": {"avrg": [],"stdev": []},
    }

    for pulse in pulses:
        print(pulse)
        reader = ST40Reader(pulse, np.min(tstart - dt*2), np.max(tend + dt *2))
        efit_time, _ = reader._get_signal("", "efit", ":time", 0)
        if np.array_equal(efit_time, "FAILED"):
            print("no Ip")
            continue

        if np.min(efit_time) > np.max(tend) or np.max(efit_time) < np.max(tstart):
            print("no Ip in time range")
            continue

        data_ip, _ = reader._get_signal("", "efit", ".constraints.ip:cvalue", 0)
        data_wp, _ = reader._get_signal("", "efit", ".virial:wp", 0)

        nirh1, nirh1_path = reader._get_signal("interferom", "nirh1", ".line_int:ne", 0)
        nirh1_time, _ = reader._get_signal_dims(nirh1_path, 1)
        nirh1_time = nirh1_time[0]

        puff, puff_path = reader._get_signal("", "gas", ".puff_valve:gas_total", -1)
        puff_time, _ = reader._get_signal_dims(puff_path, 1)
        puff_time = puff_time[0]

        brems_pi, brems_pi_path = reader._get_signal("spectrom", "princeton.passive", ".dc:brem_mp", 0)
        brems_pi_time, _ = reader._get_signal_dims(brems_pi_path, 1)
        brems_pi_time = brems_pi_time[0]

        brems_mp, brems_mp_path = reader._get_signal("spectrom", "lines", ".brem_mp1:intensity", -1)
        brems_mp_time, _ = reader._get_signal_dims(brems_mp_path, 1)
        brems_mp_time = brems_mp_time[0]

        xrcs_ti, xrcs_ti_path = reader._get_signal("sxr", "xrcs", ".te_kw:ti", 0)
        xrcs_te, xrcs_te_path = reader._get_signal("sxr", "xrcs", ".te_kw:te", 0)
        xrcs_time, _ = reader._get_signal_dims(xrcs_te_path, 1)
        xrcs_time = xrcs_time[0]

        hnbi, hnbi_path = reader._get_signal("raw_nbi", "hnbi1", ".hv_ps:i_jema", -1)
        hnbi_time, _ = reader._get_signal_dims(hnbi_path, 1)
        hnbi_time = hnbi_time[0]

        avrg_ip = []
        stdev_ip = []
        avrg_wp = []
        stdev_wp = []
        avrg_nirh1 = []
        stdev_nirh1 = []
        avrg_brems_pi = []
        stdev_brems_pi = []
        avrg_brems_mp = []
        stdev_brems_mp = []
        avrg_xrcs_te = []
        stdev_xrcs_te = []
        avrg_xrcs_ti = []
        stdev_xrcs_ti = []
        avrg_hnbi = []
        stdev_hnbi = []
        avrg_lines = []
        stdev_lines = []
        puff_sum = []
        for tind in range(np.size(tstart)):
            avrg, std = calc_mean_std(efit_time, data_ip, tstart[tind], tend[tind], upper=2.e6)
            avrg_ip.append(avrg)
            stdev_ip.append(std)

            avrg, std = calc_mean_std(efit_time, data_wp, tstart[tind], tend[tind], upper=1.e6)
            avrg_wp.append(avrg)
            stdev_wp.append(std)

            it = np.where(puff_time < t[tind] + dt)[0]
            puff_sum.append(np.sum(puff[it]) * (puff_time[1] - puff_time[0]))

            avrg, std = calc_mean_std(nirh1_time, nirh1, tstart[tind], tend[tind], upper=1.e21)
            avrg_nirh1.append(avrg)
            stdev_nirh1.append(std)

            avrg, std = calc_mean_std(brems_pi_time, brems_pi, tstart[tind], tend[tind])
            avrg_brems_pi.append(avrg)
            stdev_brems_pi.append(std)

            avrg, std = calc_mean_std(brems_mp_time, brems_mp, tstart[tind], tend[tind])
            avrg_brems_mp.append(avrg)
            stdev_brems_mp.append(std)

            avrg, std = calc_mean_std(xrcs_time, xrcs_te, tstart[tind], tend[tind])
            avrg_xrcs_te.append(avrg)
            stdev_xrcs_te.append(std)

            avrg, std = calc_mean_std(xrcs_time, xrcs_ti, tstart[tind], tend[tind])
            avrg_xrcs_ti.append(avrg)
            stdev_xrcs_ti.append(std)

            avrg, std = calc_mean_std(hnbi_time, hnbi, tstart[tind], tend[tind], toffset=-0.5)
            avrg_hnbi.append(avrg)
            stdev_hnbi.append(std)

            lines_avrg = []
            lines_stdev = []
            line_time, _ = reader._get_signal("spectrom", "avantes.line_mon", ":time", 0)
            for line in lines:
                line_data, _ = reader._get_signal(
                    "spectrom", "avantes.line_mon", f".line_evol.{line}:intensity", 0,
                )
                avrg, std = calc_mean_std(line_time, line_data, tstart[tind], tend[tind])
                lines_avrg.append(avrg)
                lines_stdev.append(std)

            avrg_lines.append(lines_avrg)
            stdev_lines.append(lines_stdev)

        params["pulses"].append(pulse)
        results["puff"].append(puff_sum)
        results["ipla"]["avrg"].append(avrg_ip)
        results["ipla"]["stdev"].append(stdev_ip)
        results["wp"]["avrg"].append(avrg_wp)
        results["wp"]["stdev"].append(stdev_wp)
        results["nirh1"]["avrg"].append(avrg_nirh1)
        results["nirh1"]["stdev"].append(stdev_nirh1)
        results["brems_pi"]["avrg"].append(avrg_brems_pi)
        results["brems_pi"]["stdev"].append(stdev_brems_pi)
        results["brems_mp"]["avrg"].append(avrg_brems_mp)
        results["brems_mp"]["stdev"].append(stdev_brems_mp)
        results["xrcs_te"]["avrg"].append(avrg_xrcs_te)
        results["xrcs_te"]["stdev"].append(stdev_xrcs_te)
        results["xrcs_ti"]["avrg"].append(avrg_xrcs_ti)
        results["xrcs_ti"]["stdev"].append(stdev_xrcs_ti)
        results["nbi"]["avrg"].append(avrg_hnbi)
        results["nbi"]["stdev"].append(stdev_hnbi)
        results["lines"]["avrg"].append(avrg_lines)
        results["lines"]["stdev"].append(stdev_lines)

    for k1 in results.keys():
        if type(results[k1]) == dict:
            for k2 in results[k1].keys():
                results[k1][k2] = np.array(results[k1][k2])
        else:
            results[k1] = np.array(results[k1])

    return params, results

def calc_mean_std(time, data, tstart, tend, lower=0., upper=None, toffset=None):
    avrg = np.nan
    std = np.nan
    offset = 0
    if (
            not np.array_equal(data, "FAILED")
            and (data.size == time.size)
            and data.size > 1
    ):
        it = (time >= tstart) * (time <= tend)
        if lower is not None:
            it *= (data > lower)
        if upper is not None:
            it *= (data < upper)

        it = np.where(it)[0]

        if toffset is not None:
            it_offset = np.where(time <= toffset)[0]
            if len(it_offset)>1:
                offset = np.mean(data[it_offset])

        avrg = np.mean(data[it] + offset)
        if len(it) >= 2:
            std = np.std(data[it] + offset)

    return avrg, std

def plot_trends(params, results, savefig=False, xlim=(), heat="all"):
    """

    Parameters
    ----------
    results
        Output from read_data method
    savefig
        Save figure to file
    xlim
        Set xlim to values different from extremes in database
    heating
        Restrict analysis to specific heating only
        "all", "ohmic", "nbi"

    Returns
    -------

    """

    lines = params["lines_labels"]
    elements = np.unique([l.split("_")[0] for l in lines])

    if len(xlim) == 0:
        xlim = (params["pulse_start"] - 2, params["pulse_end"] + 2)
    pulse_range = f"{xlim[0]}-{xlim[1]}"

    time_tit = []
    time_int_tit = []
    for it, t in enumerate(params["t"]):
        time_range = f"{params['tstart'][it]:1.3f}-{params['tend'][it]:1.3f}"
        time_tit.append(f"t=[{time_range}]")
        time_int_tit.append(f"t<{params['t'][it] + params['dt']:1.3f}")
    name = f"Avantes_trends_{pulse_range}"

    print(heat)
    plot_data = select_data(params, results, heat)

    plt.figure()
    for it, t in enumerate(params["t"]):
        plt.errorbar(
            params["pulses"],
            plot_data["ipla"]["avrg"][:, it] / 1.0e6,
            yerr=plot_data["ipla"]["stdev"][:, it] / 1.0e6,
            fmt="o",
            label=time_tit[it],
        )
    plt.ylim(0,)
    ylim = plt.ylim()
    for b in BORONISATION:
        plt.vlines(b, ylim[0], ylim[1])
    plt.xlim(xlim)
    plt.xlabel("Pulse #")
    plt.ylabel("$(MA)$")
    plt.legend()
    plt.title(f"Plasma current")
    if savefig:
        save_figure(fig_name=name + "_ipla")

    plt.figure()
    for it, t in enumerate(params["t"]):
        plt.errorbar(
            params["pulses"],
            plot_data["wp"]["avrg"][:, it] / 1.0e3,
            yerr=plot_data["wp"]["stdev"][:, it] / 1.0e3,
            fmt="o",
            label=time_tit[it],
        )
    plt.ylim(0,)
    ylim = plt.ylim()
    for b in BORONISATION:
        plt.vlines(b, ylim[0], ylim[1])
    plt.xlim(xlim)
    plt.xlabel("Pulse #")
    plt.ylabel("$(kJ)$")
    plt.legend()
    plt.title(f"Plasma energy")
    if savefig:
        save_figure(fig_name=name + "_wp")

    plt.figure()
    for it, t in enumerate(params["t"]):
        plt.errorbar(
            params["pulses"],
            plot_data["nirh1"]["avrg"][:, it] / 1.e19,
            yerr=plot_data["nirh1"]["stdev"][:, it] / 1.e19,
            fmt="o",
            label=time_tit[it],
        )
    plt.ylim(0,)
    ylim = plt.ylim()
    for b in BORONISATION:
        plt.vlines(b, ylim[0], ylim[1])
    plt.xlim(xlim)
    plt.xlabel("Pulse #")
    plt.ylabel("$(10^{19} m^{-3})$")
    plt.legend()
    plt.title(f"NIRH1 LOS-int density")
    if savefig:
        save_figure(fig_name=name + "_nirh1")

    plt.figure()
    for it, t in enumerate(params["t"]):
        plt.errorbar(
            params["pulses"],
            plot_data["brems_pi"]["avrg"][:, it],
            yerr=plot_data["brems_pi"]["stdev"][:, it],
            fmt="o",
            label=time_tit[it],
        )
    plt.ylim(0,)
    ylim = plt.ylim()
    for b in BORONISATION:
        plt.vlines(b, ylim[0], ylim[1])
    plt.xlim(xlim)
    plt.xlabel("Pulse #")
    plt.ylabel("$(a.u.)$")
    plt.legend()
    plt.title(f"PI LOS-int Bremsstrahlung")
    if savefig:
        save_figure(fig_name=name + "_brems_pi")

    plt.figure()
    for it, t in enumerate(params["t"]):
        plt.errorbar(
            params["pulses"],
            plot_data["brems_mp"]["avrg"][:, it],
            yerr=plot_data["brems_mp"]["stdev"][:, it],
            fmt="o",
            label=time_tit[it],
        )
    plt.ylim(0,)
    ylim = plt.ylim()
    for b in BORONISATION:
        plt.vlines(b, ylim[0], ylim[1])
    plt.xlim(xlim)
    plt.xlabel("Pulse #")
    plt.ylabel("$(a.u.)$")
    plt.legend()
    plt.title(f"MP filter LOS-int Bremsstrahlung")
    if savefig:
        save_figure(fig_name=name + "_brems_mp")

    plt.figure()
    for it, t in enumerate(params["t"]):
        plt.errorbar(
            params["pulses"],
            plot_data["xrcs_te"]["avrg"][:, it]/1.e3,
            yerr=plot_data["xrcs_te"]["stdev"][:, it]/1.e3,
            fmt="o",
            label=time_tit[it],
        )
    plt.ylim(0,)
    ylim = plt.ylim()
    for b in BORONISATION:
        plt.vlines(b, ylim[0], ylim[1])
    plt.xlim(xlim)
    plt.xlabel("Pulse #")
    plt.ylabel("$(keV)$")
    plt.legend()
    plt.title(f"XRCS Te")
    if savefig:
        save_figure(fig_name=name + "_xrcs_te")

    plt.figure()
    for it, t in enumerate(params["t"]):
        plt.errorbar(
            params["pulses"],
            plot_data["xrcs_ti"]["avrg"][:, it]/1.e3,
            yerr=plot_data["xrcs_ti"]["stdev"][:, it]/1.e3,
            fmt="o",
            label=time_tit[it],
        )
    plt.ylim(0,)
    ylim = plt.ylim()
    for b in BORONISATION:
        plt.vlines(b, ylim[0], ylim[1])
    plt.xlim(xlim)
    plt.xlabel("Pulse #")
    plt.ylabel("$(keV)$")
    plt.legend()
    plt.title(f"XRCS Ti")
    if savefig:
        save_figure(fig_name=name + "_xrcs_ti")

    plt.figure()
    for it, t in enumerate(params["t"]):
        plt.errorbar(
            params["pulses"],
            plot_data["nbi"]["avrg"][:, it],
            yerr=plot_data["nbi"]["stdev"][:, it],
            fmt="o",
            label=time_tit[it],
        )
    plt.ylim(0,)
    ylim = plt.ylim()
    for b in BORONISATION:
        plt.vlines(b, ylim[0], ylim[1])
    plt.xlim(xlim)
    plt.xlabel("Pulse #")
    plt.ylabel("$(a.u)$")
    plt.legend()
    plt.title(f"NBI power")
    if savefig:
        save_figure(fig_name=name + "_hnbi")

    plt.figure()
    for it, t in enumerate(params["t"]):
        plt.scatter(
            params["pulses"], plot_data["puff"][:, it], marker="o", label=time_int_tit[it],
        )
    plt.ylim(0,)
    ylim = plt.ylim()
    for b in BORONISATION:
        plt.vlines(b, ylim[0], ylim[1])
    plt.xlim(xlim)
    plt.xlabel("Pulse #")
    plt.ylabel("$(V * s)$")
    plt.legend()
    plt.title(f"Total gas puff")
    if savefig:
        save_figure(fig_name=name + "_puff")

    for elem in elements:
        print(elem)
        plt.figure()
        elem_name = elem.upper()
        if len(elem) > 1:
            elem_name = elem_name[0] + elem_name[1].lower()
        ymax = 0.0
        for it, t in enumerate(params["t"]):
            for il, l in enumerate(lines):
                lsplit = l.split("_")
                if elem != lsplit[0]:
                    continue

                label = f"{elem_name}{lsplit[1].upper()} {lsplit[2]} nm " + time_tit[it]
                y = plot_data["lines"]["avrg"][:, it, il]
                yerr = plot_data["lines"]["stdev"][:, it, il]
                ymax = np.max([ymax, np.max(y + yerr)])
                plt.errorbar(
                    params["pulses"], y, yerr=yerr, fmt="o", label=label,
                )
        plt.xlim(xlim)
        plt.ylim(0,)
        ylim = plt.ylim()
        for b in BORONISATION:
            plt.vlines(b, ylim[0], ylim[1])
        plt.legend()
        plt.xlabel("Pulse #")
        plt.ylabel("Intensity")
        plt.title(f"{elem_name} lines")
        if savefig:
            save_figure(fig_name=name + f"_{elem_name}_lines")

def select_data(params, results, heat):
    temporary = deepcopy(results)
    npulses = np.size(params["pulses"])
    nan_array = np.array([np.nan] * npulses)

    hind = {"all":[], "nbi":[], "ohmic":[]}
    for it, t in enumerate(params["t"]):
        hind["all"] = np.array([True]*len(params["pulses"]))
        nbi_ind = (results["nbi"]["avrg"][:, it] > 0)
        hind["ohmic"] = (nbi_ind == False)
        hind["nbi"] = (nbi_ind == True)

        for k1 in temporary.keys():
            if type(temporary[k1]) == dict:
                for k2 in temporary[k1].keys():
                    if len(temporary[k1][k2].shape) == 2:
                        tmp = temporary[k1][k2][:, it]
                        temporary[k1][k2][:, it] = np.where(hind[heat], tmp, nan_array)
                    else:
                        for il in range(len(temporary[k1][k2][0, it, :])):
                            tmp = temporary[k1][k2][:, it, il]
                            temporary[k1][k2][:, it, il] = np.where(hind[heat], tmp, nan_array)
            else:
                tmp = temporary[k1][:, it]
                temporary[k1][:, it] = np.where(hind[heat], tmp, [np.nan]*npulses)

    return temporary

def save_figure(fig_name="", orientation="landscape", ext=".jpg"):
    plt.savefig(
        f"/home/{getpass.getuser()}/python/figures/Avantes_trends/" + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )
