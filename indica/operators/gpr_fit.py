from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
import xarray as xr
from xarray import DataArray

from indica.readers.read_st40 import ReadST40
from indica.utilities import FIG_PATH
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.utilities import set_plot_rcparams

CMAP, COLORS = set_plot_colors()


def gpr_fit(
    x: np.array,
    y: np.array,
    y_err: np.array,
    x_fit: np.array,
    kernel: kernels.Kernel,
):
    """
    Run GPR fit given input data and desired x-grid

    Parameters
    ----------
    x
        x-axis of the data
    y
        data value
    y_err
        data error
    x_fit
        x-axis for fitting
    kernel
        Kernel used

    Returns
    -------
    Fit and error on desired x-grid

    """

    isort = np.argsort(x)
    _x = np.sort(x)
    _y = np.interp(_x, x[isort], y[isort])
    _y_err = np.interp(_x, x[isort], y_err[isort])

    idx = np.isfinite(_y)

    _x = _x[idx].reshape(-1, 1)
    _y_err = _y_err[idx]
    _y = _y[idx].reshape(-1, 1)

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=_y_err**2,
    )  # alpha is sigma^2
    gpr.fit(
        _x,
        _y,
    )

    _x_fit = x_fit.reshape(-1, 1)
    y_fit, y_fit_err = gpr.predict(_x_fit, return_std=True)

    return y_fit, y_fit_err, gpr


def plot_gpr_fit(
    data: DataArray,
    y_fit: DataArray,
    y_fit_err: DataArray,
    fig_style: str = "profiles",
    ylabel: str = "",
    xlabel: str = "",
    label: str = "",
    title: str = "",
    fig_name: str = "",
    save_fig: bool = False,
    color=None,
):
    set_plot_rcparams(fig_style)

    plt.figure()

    x_data = getattr(data, data.dims[0])
    x_fit = getattr(y_fit, data.dims[0])
    y_err = data.error

    plt.fill_between(
        x_fit,
        y_fit - y_fit_err,
        y_fit + y_fit_err,
        alpha=0.5,
        color=color,
    )
    plt.plot(x_fit, y_fit, color=color)
    plt.errorbar(x_data, data, y_err, linestyle="", color=color)
    plt.plot(x_data, data, marker="o", linestyle="", color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    set_axis_sci()
    if len(label) > 0:
        plt.legend()

    if save_fig:
        plt.savefig(
            FIG_PATH + f"{fig_name}_GPR_t:{data.t.values:.3f}.png", bbox_inches="tight"
        )


def post_process_ts(binned_data, equilibrium, quant, pulse, split=""):
    rmag = binned_data["efit"]["rmag"]
    data = binned_data["ts"][quant]
    data["pulse"] = pulse
    data["quantity"] = quant

    data.transform.set_equilibrium(equilibrium)
    data.transform.convert_to_rho_theta(t=data.t)
    data["rho"] = data.transform.rho

    # Normalising
    if quant == "ne":
        data.values = data.values * 1e-19
        data["error"] = data.error * 1e-19
        data["unit"] = "n19"
    elif quant == "te":
        data.values = data.values * 1e-3
        data["error"] = data.error * 1e-3
        data["unit"] = "keV"
    else:
        raise ValueError(f"Unknown data quantity: {quant}")

    rmag = rmag[0]
    if split == "HFS":
        data = data.where(data.R < rmag)
    elif split == "LFS":
        data = data.where(data.R > rmag)
    else:
        data = data

    return data


def gpr_fit_ts(
    data: xr.DataArray,
    kernel: kernels.Kernel,
    xdim: str = "rho",
    virtual_obs=True,
    virtual_symmetry=True,
    x_bounds=None,
    virtual_points=None,
    plot=False,
    save_fig=False,
):
    if x_bounds is None:
        if xdim == "rho":
            x_bounds = (0, 1.3)
        elif xdim == "R":
            x_bounds = (0.1, 0.9)

    if virtual_points is None:
        if xdim == "rho":
            virtual_points = [
                (1.5, lambda y: 0),
            ]
        elif xdim == "R":
            virtual_points = [(0, lambda y: 0), (0.9, lambda y: 0)]

    x_fit = np.linspace(x_bounds[0], x_bounds[1], 1000)
    y_fit = []
    y_fit_err = []
    gpr = []

    for t in data.t:
        if "t" in data.__getattr__(xdim).dims:
            x = data.__getattr__(xdim).sel(t=t).values
        else:
            x = data.__getattr__(xdim).values
        y = data.sel(t=t).values
        y_err = data.error.sel(t=t).values

        if virtual_obs:
            num_vo = len(virtual_points)
            x = np.insert(
                x,
                [i for i in range(num_vo)],
                [virtual_point[0] for virtual_point in virtual_points],
            )
            y = np.insert(
                y,
                [i for i in range(num_vo)],
                [virtual_point[1](y) for virtual_point in virtual_points],
            )
            y_err = np.insert(
                y_err, [i for i in range(num_vo)], [0.01 for i in range(num_vo)]
            )

            if virtual_symmetry:
                x = np.concatenate((x, -x))
                y = np.concatenate((y, y))
                y_err = np.concatenate((y_err, y_err))

        _y_fit, _y_fit_err, _gpr = gpr_fit(
            x,
            y,
            y_err,
            x_fit,
            kernel,
        )
        y_fit.append(DataArray(_y_fit, coords=[(xdim, x_fit)]))
        y_fit_err.append(DataArray(_y_fit_err, coords=[(xdim, x_fit)]))
        gpr.append(_gpr)

    fit = xr.concat(y_fit, "t").assign_coords(t=data.t)
    fit_err = xr.concat(y_fit_err, "t").assign_coords(t=data.t)

    if plot or save_fig:
        Path(FIG_PATH).mkdir(parents=True, exist_ok=True)
        for idx, tplot in enumerate(data.t.values):
            plot_gpr_fit(
                data.sel(t=tplot).swap_dims({"channel": xdim}),
                fit.sel(t=tplot),
                fit_err.sel(t=tplot),
                ylabel=f"{data.quantity.values} ({data.unit.values})",
                xlabel=f"{xdim}",
                title=f"{data.pulse.values}\nOptimimum: {gpr[idx].kernel_}",
                fig_name=f"{data.pulse.values}_TS.{data.quantity.values}",
                save_fig=save_fig,
            )

            if not save_fig:
                plt.show()

    plt.close("all")
    return fit, fit_err


if __name__ == "__main__":
    pulse = 11089
    tstart = 0.03
    tend = 0.15
    dt = 0.01
    st40 = ReadST40(pulse, tstart, tend, dt)
    st40.get_equilibrium()
    st40(
        instruments=[
            "ts",
            "efit",
        ],
        R_shift=xr.full_like(st40.equilibrium.t, 0.02),
    )

    ne_data = post_process_ts(
        st40.binned_data,
        st40.equilibrium,
        "ne",
        pulse,
        split="",
    )
    ne_kernel = 1.0 * kernels.RationalQuadratic(
        alpha_bounds=(0.1, 1.0), length_scale_bounds=(0.3, 0.7)
    ) + kernels.WhiteKernel(noise_level_bounds=(0.01, 10))

    te_data = post_process_ts(
        st40.binned_data,
        st40.equilibrium,
        "te",
        pulse,
        split="",
    )
    te_kernel = 1.0 * kernels.RationalQuadratic(
        alpha_bounds=(0.1, 1.0), length_scale_bounds=(0.4, 0.7)
    ) + kernels.WhiteKernel(noise_level_bounds=(0.01, 10))

    gpr_fit_ts(
        data=ne_data,
        xdim="rho",
        virtual_obs=True,
        virtual_symmetry=True,
        kernel=ne_kernel,
        save_fig=True,
    )
    gpr_fit_ts(
        data=te_data,
        xdim="rho",
        virtual_obs=True,
        virtual_symmetry=True,
        kernel=te_kernel,
        save_fig=True,
    )
