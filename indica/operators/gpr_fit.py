import getpass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
import xarray as xr
from xarray import DataArray

from indica.readers.read_st40 import ReadST40
from indica.utilities import save_figure
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.utilities import set_plot_rcparams

FIG_PATH = f"/home/{getpass.getuser()}/figures/Indica/profile_fits/"
CMAP, COLORS = set_plot_colors()


def choose_kernel(kernel: str):
    """
    Wrapper to return GPR Kernel function selected by name

    Parameters
    ----------
    kernel
        Name of kernel

    Returns
    -------
        Kernel function for GPR call

    """
    kernels = {
        "RBF_noise": RadialBasisFunction_WithNoise(),
        "RBF": RadialBasisFunction_NoNoise(),
    }

    if kernel in kernels.keys():
        return kernels[kernel]
    else:
        raise ValueError


def RadialBasisFunction_NoNoise(length_scale=0.2, dlength_scale=0.001, **kwargs):
    kernel = 1.0 * kernels.RBF(
        length_scale=length_scale,
        length_scale_bounds=(
            length_scale - dlength_scale,
            length_scale + dlength_scale,
        ),
    )
    return kernel


def RadialBasisFunction_WithNoise(
    length_scale=0.1, dlength_scale=0.001, noise_level=1000, dnoise_level=3000, **kwargs
):
    kernel = 1.0 * kernels.RBF(
        length_scale=length_scale,
        length_scale_bounds=(
            length_scale - dlength_scale,
            length_scale + dlength_scale,
        ),
    ) + kernels.WhiteKernel(
        noise_level=noise_level,
        noise_level_bounds=(noise_level - dnoise_level, noise_level + dnoise_level),
    )
    return kernel


def gpr_fit(
    x: np.array,
    y: np.array,
    y_err: np.array,
    x_fit: np.array,
    kernel_name: str = "RBF_noise",
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
        Kernel name

    Returns
    -------
    Fit and error on desired x-grid

    """

    kernel = choose_kernel(kernel_name)

    isort = np.argsort(x)
    _x = np.sort(x)
    _y = np.interp(_x, x[isort], y[isort])
    _y_err = np.interp(_x, x[isort], y_err[isort])

    idx = np.isfinite(_y)

    _x = _x[idx].reshape(-1, 1)
    _y_err = _y_err[idx]
    _y = _y[idx].reshape(-1, 1)

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=_y_err**2,)  # alpha is sigma^2
    gpr.fit(_x, _y, )

    _x_fit = x_fit.reshape(-1, 1)
    y_fit, y_fit_err = gpr.predict(_x_fit, return_std=True)

    return y_fit, y_fit_err


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
    y_err = data.error.sel(t=data.t)

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
    plt.title(f"{title} t = {data.t.values:.3f} s")
    set_axis_sci()
    if len(label) > 0:
        plt.legend()

    if save_fig:
        save_figure(FIG_PATH, f"{fig_name}_GPR_fit_{data.t:.3f}_s", save_fig=save_fig)


def example_run(
    pulse: int = 10619,
    tstart=0.06,
    tend=0.1,
    kernel_name: str = "RBF_noise",
    plot=True,
    save_fig=False,
    xdim: str = "R",
    split="LFS",
    virtual_obs=True,

):

    st40 = ReadST40(pulse, tstart, tend)
    st40(instruments=["ts", "efit"])
    rmag = st40.binned_data["efit"]["rmag"]

    data = st40.raw_data["ts"]["te"]
    data.transform.set_equilibrium(st40.equilibrium)
    data.transform.convert_to_rho_theta(t=data.t)
    data["rho"] = data.transform.rho

    if xdim == "R":
        x_bounds = data.transform._machine_dims[0]
    else:
        x_bounds = (0, 1.2)

    data = data[(data.t >= tstart) & (data.t <= tend)]
    rmag = rmag[0]

    if split == "HFS":
        data = data.where(data.R < rmag)
    elif split == "LFS":
        data = data.where(data.R > rmag)
    else:
        data = data

    x_fit = np.linspace(x_bounds[0], x_bounds[1], 1000)
    dx = x_fit[1] - x_fit[0]
    y_fit = []
    y_fit_err = []
    for t in data.t:
        if "t" in data.__getattr__(xdim).dims:
            x = data.__getattr__(xdim).sel(t=t).values
        else:
            x = data.__getattr__(xdim).values
        y = data.sel(t=t).values
        y_err = data.error.sel(t=t).values

        if virtual_obs:
            x = np.insert(x, [0, x.size], [x_bounds[0], x_bounds[1]])
            if xdim == "rho":
                y = np.insert(y, [0, y.size], [np.nanmax(y), 0])
            else:
                y = np.insert(y, [0, y.size], [0, 0])

            y_err = np.insert(y_err, [0, y_err.size], [1, 1])

        _y_fit, _y_fit_err = gpr_fit(
            x,
            y,
            y_err,
            x_fit,
            kernel_name=kernel_name,
        )
        y_fit.append(DataArray(_y_fit, coords=[(xdim, x_fit)]))
        y_fit_err.append(DataArray(_y_fit_err, coords=[(xdim, x_fit)]))

    fit = xr.concat(y_fit, "t").assign_coords(t=data.t)
    fit_err = xr.concat(y_fit_err, "t").assign_coords(t=data.t)

    if plot or save_fig:
        plt.ioff()
        fig_name = f"{pulse}_TS_Te"
        for tplot in data.t.values:
            plot_gpr_fit(
                data.sel(t=tplot).swap_dims({"channel":xdim}),
                fit.sel(t=tplot),
                fit_err.sel(t=tplot),
                ylabel="Te [eV]",
                xlabel=f"{xdim}",
                title=str(st40.pulse),
                fig_name=f"{fig_name}_vs_R",
                save_fig=save_fig,
            )

            if not save_fig:
                plt.show()

    return data, fit, fit_err


if __name__ == "__main__":

    # example_run(pulse=11314, xdim="rho", split="HFS", virtual_obs=True, kernel_name="RBF")
    example_run(pulse=11314, xdim="rho", split="LFS", virtual_obs=True, kernel_name="RBF")
