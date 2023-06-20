import getpass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from indica.numpy_typing import ArrayLike
from indica.readers.read_st40 import ReadST40
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.utilities import set_plot_rcparams

FIG_PATH = f"/home/{getpass.getuser()}/figures/Indica/time_evolution/"
CMAP, COLORS = set_plot_colors()
LINESTYLES = ["solid", "dashed", "dotted"]
set_plot_rcparams("profiles")


def plot_sxrc(
    pulse: int = 10607,  # 10605
    what_to_read: list = ["sxrc_xy2:brightness"],
    tplot: ArrayLike = [0.067, 0.069],  # [0.07, 0.09]
    save_fig=False,
    plot_raw: bool = False,
    xvar: str = "t",
    yvar: str = "channel",
    tstart=0.02,
    tend=0.1,
    dt: float = 0.001,
    data_key="binned_data",
):

    instruments = []
    quantities = []
    linestyles = {}
    for i, identifier in enumerate(what_to_read):
        instr, quant = identifier.split(":")
        instruments.append(instr)
        quantities.append(quant)
        linestyles[instr] = LINESTYLES[i]

    st40 = ReadST40(pulse, tstart, tend, dt=dt)
    st40(instruments=instruments)

    data: dict = {}
    for instr, quant in zip(instruments, quantities):
        if instr not in data.keys():
            data[instr] = {}
        data[instr][quant] = getattr(st40, data_key)[instr][quant].transpose(yvar, xvar)
        _rho, _theta = data[instr][quant].transform.convert_to_rho_theta(
            t=data[instr][quant].t
        )

    for instr, quant in zip(instruments, quantities):
        plt.figure()
        plot = data[instr][quant].plot(label=f"{instr}:{quant}")
        set_axis_sci(plot_object=plot)
        plt.title(f"{instr.upper()} for pulse {pulse}")

    if tplot is not None:
        cols = CMAP(np.linspace(0.75, 0.1, np.size(tplot), dtype=float))

        plt.figure()
        for icol, t in enumerate(tplot):
            for instr, quant in zip(instruments, quantities):
                _data = data[instr][quant].sel(t=t, method="nearest")
                if "error" in _data.attrs:
                    _err = (_data.error + _data.stdev).sel(t=t, method="nearest")
                else:
                    _err = xr.full_like(_data, 0.0)
                _R_diff = (
                    _data.transform.impact_parameter.value
                    - _data.transform.equilibrium.rmag
                ).sel(t=t, method="nearest")
                plt.fill_between(
                    _R_diff.values,
                    _data.values + _err.values,
                    _data.values - _err.values,
                    color=cols[icol],
                    alpha=0.5,
                )
                plt.plot(
                    _R_diff.values,
                    _data.values,
                    label=f"t={t:.3f}",
                    linestyle=linestyles[instr],
                    color=cols[icol],
                )
        plt.xlabel("Impact R - R$_{mag}$")
        plt.ylabel(f"{_data.long_name} [{_data.units}]")
        plt.title(f"{instr.upper()} for pulse {pulse}")
        plt.legend()
        set_axis_sci()

    return st40


# def plot_time_surface(
#     pulse: int = 10605,
#     instruments: list = ["sxrc_xy1"],
#     quantity: str = "brightness",
#     tplot: ArrayLike = None,
#     save_fig=False,
#     plot_raw: bool = False,
#     xvar: str = "time",
#     yvar: str = "channel",
# ):

if __name__ == "__main__":
    plt.ioff()
    plot_sxrc()
    plt.show()
