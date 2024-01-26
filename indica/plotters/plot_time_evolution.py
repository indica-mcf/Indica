from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from indica.readers.read_st40 import ReadST40
from indica.utilities import FIG_PATH
from indica.utilities import save_figure
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.utilities import set_plot_rcparams

CMAP, COLORS = set_plot_colors()
LINESTYLES = ["solid", "dashed", "dotted"]
set_plot_rcparams("profiles")
QUANTITIES: list = [
    "smmh:ne",
    "xrcs:ti_w",
    "xrcs:te_n3w",
    "xrcs:te_kw",
    "xrcs:spectra",
    "sxrc_xy2:brightness",
    "efit:ipla",
    "efit:rmag",
    "efit:zmag",
    "sxr_spd:brightness",
    # "cxff_pi:ti",
    # "cxff_pi:vtor",
    # "cxff_tws_c:ti",
    # "cxff_tws_c:vtor",
    "ts:te",
    "ts:ne",
    # "lines",
]

Y0 = {}
Y0["nirh1"] = True
Y0["smmh1"] = True
Y0["xrcs"] = True
Y0["sxr_diode_1"] = True
Y0["brems"] = True
Y0["efit"] = True
Y0["cxff_tws_c"] = True
Y0["cxff_pi"] = True


def plot_st40_data(
    pulses: list,
    tstart: float = 0.02,
    tend: float = 0.1,
    dt: float = 0.005,
    quantities: list = [],
    tplot: float = None,
    save_fig: bool = False,
    fig_path: str = None,
    fig_style: str = "profiles",
    plot_binned: bool = True,
):

    if tplot is None:
        tplot = np.mean([tstart, tend])
    if len(quantities) == 0:
        quantities = QUANTITIES
    instruments = list(np.unique([quant.split(":")[0] for quant in quantities]))
    if fig_path is None:
        fig_path = f"{FIG_PATH}"

    set_plot_rcparams(fig_style)
    xr.set_options(keep_attrs=True)
    if len(pulses) > 1:
        colors = CMAP(np.linspace(0.75, 0.1, len(pulses), dtype=float))
    else:
        colors = ["blue"]

    raw: dict = {quant: {} for quant in quantities}
    binned = deepcopy(raw)
    data = {"raw": raw, "binned": binned}
    for i, pulse in enumerate(pulses):
        fig_path += f"_{pulse}"
        st40 = ReadST40(pulse, tstart=tstart, tend=tend, dt=dt)
        st40(instruments)
        for quantity in quantities:
            instr, quant = quantity.split(":")
            data["binned"][quantity][pulse] = st40.binned_data[instr][quant]
            data["raw"][quantity][pulse] = st40.raw_data[instr][quant]

    for quantity in quantities:
        print(quantity)
        instr, _ = quantity.split(":")
        plt.figure()
        for i, pulse in enumerate(pulses):
            color = colors[i]
            if len(data["raw"][quantity].keys()) == 0:
                continue
            plot_data(data, quantity, pulse, tplot, key="raw", color=color)
            if plot_binned:
                plot_data(data, quantity, pulse, tplot, key="binned", color=color)

            set_axis_sci()
            if instr in Y0.keys():
                plt.ylim(
                    0,
                )
            plt.legend()
            plt.autoscale()
            save_figure(fig_path, f"{quantity}", save_fig=save_fig)

    return data, st40


def plot_data(data, quantity: str, pulse: int, tplot: float, key="raw", color=None):
    str_to_add = ""
    instr, quant = quantity.split(":")
    if key == "raw":
        marker = None
    else:
        marker = "o"

    _data = data[key][quantity][pulse]
    tslice = slice(_data.t.min().values, _data.t.max().values)
    if "error" not in _data.attrs:
        _data.attrs["error"] = xr.full_like(_data, 0.0)
    if "stdev" not in _data.attrs:
        _data.attrs["stdev"] = xr.full_like(_data, 0.0)
    _err = np.sqrt(_data.error**2 + _data.stdev**2)
    _err = xr.where(_err / _data.values < 1.0, _err, 0.0)
    if len(_data.dims) > 1:
        str_to_add = f" @ {tplot:.3f} s"
        tslice = _data.t.sel(t=tplot, method="nearest")

    _data = _data.sel(t=tslice)
    _err = _err.sel(t=tslice)
    if instr in "xrcs" and quant == "spectra":
        bgnd = _data.sel(wavelength=slice(0.393, 0.388)).mean("wavelength")
        _data -= bgnd
    label = None
    if key == "raw":
        label = str(pulse)
        alpha = 0.5
        plt.fill_between(
            _data.coords[_data.dims[0]].values,
            _data.values - _err.values,
            _data.values + _err.values,
            color=color,
            alpha=alpha,
        )
    if key == "binned":
        alpha = 0.8
        plt.errorbar(
            _data.coords[_data.dims[0]].values,
            _data.values,
            _err.values,
            color=color,
            alpha=alpha,
        )
    _data.plot(label=label, color=color, alpha=alpha, marker=marker)
    plt.title(f"{instr.upper()} {quant}" + str_to_add)


if __name__ == "__main__":
    plt.ioff()
    plot_st40_data([11226, 11225])
    plt.show()
