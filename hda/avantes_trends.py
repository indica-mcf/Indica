"""Various miscellaneous functions for handling of atomic data."""

import numpy as np
import xarray as xr
from xarray import DataArray
from indica.readers import ST40Reader
import matplotlib.pylab as plt

plt.ion()


def read_data(
    pulse_start, pulse_end, t=0.03, dt=0.01, debug=False,
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

    tstart = t - dt
    tend = t + dt
    pulses = np.arange(pulse_start, pulse_end)

    lines = [
        "h_i_656",
        "he_i_588",
        "he_ii_469",
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

    pulse_list = []
    ipla_avrg = []
    ipla_stdev = []
    lines_all_avrg = []
    lines_all_stdev = []
    for pulse in pulses:
        print(pulse)
        reader = ST40Reader(pulse, 0.0, 0.1)
        time_ip, _ = reader._get_signal("", "efit", ":time", 0)
        data_ip, _ = reader._get_signal("", "efit", ".constraints.ip:cvalue", 0)
        if (
            np.array_equal(time_ip, "FAILED")
            or not all(np.isfinite(data_ip))
            or np.min(time_ip) > tend
            or np.max(time_ip) < tstart
        ):
            continue
        tmp = DataArray(data_ip, dims=("t",), coords={"t": time_ip}).sel(
            t=slice(tstart, tend)
        )
        if debug:
            plt.close("all")
            plt.figure()
            tmp.plot()
            plt.title("Ip")

        tmp = tmp[np.where(tmp > 0)[0]]
        avrg_ip = tmp.mean().values
        stdev_ip = tmp.std().values

        lines_avrg = []
        lines_stdev = []
        time_line, _ = reader._get_signal("spectrom", "avantes.line_mon", ":time", 0)
        if np.array_equal(time_line, "FAILED"):
            continue
        for line in lines:
            data_line, _ = reader._get_signal(
                "spectrom", "avantes.line_mon", f".line_evol.{line}:intensity", 0,
            )
            if (
                np.array_equal(data_line, "FAILED")
                or not all(np.isfinite(data_line))
                or np.min(time_line) > tend
                or np.max(time_line) < tstart
            ):
                continue

            tmp = DataArray(data_line, dims=("t",), coords={"t": time_line}).sel(
                t=slice(tstart, tend)
            )
            if debug:
                plt.figure()
                tmp.plot()
                plt.title(line)
                input("...continue...")

            innul = np.where(tmp > 0)[0]
            if len(innul) > 1:
                avrg = tmp.mean().values
                stdev = tmp.std().values
            else:
                avrg = np.nan
                stdev = np.nan

            lines_avrg.append(avrg)
            lines_stdev.append(stdev)

        if len(lines_avrg) > 0:
            print("  Ok")
            pulse_list.append(pulse)
            ipla_avrg.append(avrg_ip)
            ipla_stdev.append(stdev_ip)
            lines_all_avrg.append(lines_avrg)
            lines_all_stdev.append(lines_stdev)

    results = {
        "pulse_start": pulse_start,
        "pulse_end": pulse_end,
        "tstart": tstart,
        "tend": tend,
        "pulses": np.array(pulse_list),
        "ipla": {"avrg": np.array(ipla_avrg), "stdev": np.array(ipla_stdev)},
        "lines": {
            "avrg": np.array(lines_all_avrg),
            "stdev": np.array(lines_all_stdev),
            "labels": lines,
        },
    }

    return results


def plot_trends(results, savefig=False):

    lines = results["lines"]["labels"]
    elements = np.unique([l.split("_")[0] for l in lines])

    xlim = (results['pulse_start'], results['pulse_end'])
    pulse_range = f"{xlim[0]}-{xlim[1]}"
    time_range = f"{results['tstart']:1.3f}-{results['tend']:1.3f}"
    time_tit = f"t=[{time_range}]"
    name = f"Avantes_trends_{pulse_range}_t_{time_range}"
    plt.figure()
    plt.errorbar(
        results["pulses"],
        results["ipla"]["avrg"] / 1.0e6,
        yerr=results["ipla"]["stdev"] / 1.0e6,
        fmt="o",
    )
    plt.xlim(xlim)
    plt.xlabel("Pulse #")
    plt.ylabel("$I_P (MA)$")
    plt.title(f"Plasma current {time_tit}")
    if savefig:
        save_figure(fig_name=name + "_ipla")

    for elem in elements:
        plt.figure()
        elem_name = elem.upper()
        if len(elem) > 1:
            elem_name = elem_name[0] + elem_name[1].lower()
        for i, l in enumerate(lines):
            lsplit = l.split("_")
            if elem != lsplit[0]:
                continue

            label = f"{elem_name}{lsplit[1].upper()} {lsplit[2]} nm"
            y = results["lines"]["avrg"][:, i]
            yerr = results["lines"]["stdev"][:, i]
            plt.errorbar(
                results["pulses"], y, yerr=yerr, fmt="o", label=label,
            )
        plt.xlim(xlim)
        plt.legend()
        plt.xlabel("Pulse #")
        plt.ylabel("Intensity")
        plt.title(f"{elem_name} lines {time_tit}")
        if savefig:
            save_figure(fig_name=name + f"_{elem_name}_lines")

def save_figure(path_name="/home/marco.sertoli/", fig_name="", orientation="landscape", ext=".jpg"):
    plt.savefig(
        path_name + "python/figures/" + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )
