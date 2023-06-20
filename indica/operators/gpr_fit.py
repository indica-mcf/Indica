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
        "RBF_noise": RadialBasisFunction_NoNoise(),
        "RBF": RadialBasisFunction_WithNoise(),
    }

    if kernel in kernels.keys():
        return kernels[kernel]
    else:
        raise ValueError


def RadialBasisFunction_NoNoise(length_scale=0.1, dlength_scale=0.001, **kwargs):
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
    y_bounds: tuple = (1, 1),
    err_bounds: tuple = (0, 0),
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
    y_bounds
        Boundery value at the edge of the x-grid
    err_bounds
        Boundery error at the edge of the x-grid

    Returns
    -------
    Fit and error on desired x-grid

    """

    kernel = choose_kernel(kernel_name)

    isort = np.argsort(x)
    _x = np.sort(np.append(x, [x_fit[0], x_fit[-1]]))
    _y = np.interp(_x, x[isort], y[isort])

    _y[0] = y_bounds[0]
    _y[-1] = y_bounds[1]
    _y_err = np.interp(_x, x[isort], y_err[isort])
    _y_err[0] = err_bounds[0]
    _y_err[-1] = err_bounds[1]

    idx = np.isfinite(_y)

    _x = _x[idx].reshape(-1, 1)
    _y_err = _y_err[idx]
    _y = _y[idx].reshape(-1, 1)

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=_y_err**2)  # alpha is sigma^2
    gpr.fit(_x, _y)

    _x_fit = x_fit.reshape(-1, 1)
    y_fit, y_fit_err = gpr.predict(_x_fit, return_std=True)

    return y_fit, y_fit_err


def run_gpr_fit(
    data: DataArray,
    xdim: str = "R",
    kernel_name: str = "RBF_noise",
    x_bounds: tuple = (0, 1),
    y_bounds: tuple = (1, 1),
    err_bounds: tuple = (0, 0),
):
    """
    Run GPR fit for experimental Indica-native DataArray structures

    Parameters
    ----------
    data
        Data to be fitted dims = (x, t)
    xdim
        Dimension on which fit is to be performed
    kernel_name
        Name of kernel to be used for fitting

    Returns
    -------

    """

    x_fit = np.linspace(x_bounds[0], x_bounds[1], 1000)

    y_fit = []
    y_fit_err = []
    for t in data.t:
        x = data.R.values
        y = data.sel(t=t).values
        y_err = data.error.sel(t=t).values
        _y_fit, _y_fit_err = gpr_fit(
            x,
            y,
            y_err,
            x_fit,
            kernel_name=kernel_name,
            y_bounds=y_bounds,
            err_bounds=err_bounds,
        )

        y_fit.append(DataArray(_y_fit, coords=[(xdim, x_fit)]))
        y_fit_err.append(DataArray(_y_fit_err, coords=[(xdim, x_fit)]))

    fit = xr.concat(y_fit, "t").assign_coords(t=data.t)
    fit_err = xr.concat(y_fit_err, "t").assign_coords(t=data.t)

    return fit, fit_err


def plot_gpr_fit(
    data: DataArray,
    fit: DataArray,
    fit_err: DataArray,
    tplot: float,
    x_data: DataArray = None,
    x_fit: DataArray = None,
    fig_style: str = "profiles",
    ylabel: str = "",
    xlabel: str = "",
    label: str = "",
    title: str = "",
    fig_name: str = "",
    figure: bool = True,
    save_fig: bool = False,
    color=None,
):
    set_plot_rcparams(fig_style)

    if figure:
        plt.figure()

    if x_data is None:
        xdim = [dim for dim in data.dims if dim != "t"][0]
        x_data = getattr(data, xdim)

    if x_fit is None:
        xdim = [dim for dim in fit.dims if dim != "t"][0]
        x_fit = getattr(fit, xdim)

    if "t" in x_data.dims:
        _x_data = x_data.sel(t=tplot)
    else:
        _x_data = x_data

    if "t" in x_fit.dims:
        _x_fit = x_fit.sel(t=tplot)
    else:
        _x_fit = x_fit

    y_data = data.sel(t=tplot)
    y_err = data.error.sel(t=tplot)

    y_fit = fit.sel(t=tplot)
    y_fit_err = fit_err.sel(t=tplot)

    plt.fill_between(
        _x_fit,
        y_fit - y_fit_err,
        y_fit + y_fit_err,
        alpha=0.5,
        color=color,
    )
    plt.plot(_x_fit, y_fit, color=color)
    plt.errorbar(_x_data, y_data, y_err, linestyle="", color=color)
    plt.plot(_x_data, y_data, marker="o", linestyle="", color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} t = {tplot:.3f} s")
    set_axis_sci()
    if len(label) > 0:
        plt.legend()

    if save_fig:
        save_figure(FIG_PATH, f"{fig_name}_GPR_fit_{tplot:.3f}_s", save_fig=save_fig)


def example_run(
    pulse: int = 10619,
    kernel_name: str = "RBF_noise",
    plot=True,
    save_fig=False,
    xdim: str = "R",
    x_bounds: tuple = (0, 1),
    y_bounds: tuple = (1, 1),
    err_bounds: tuple = (0, 0),
):

    tstart = 0.02
    tend = 0.10

    st40 = ReadST40(pulse, tstart, tend)
    st40(instruments=["ts", "efit"])

    data = st40.raw_data["ts"]["te"]
    if xdim not in data.dims and hasattr(data, xdim):
        data = data.swap_dims({"channel": xdim})
    if xdim == "R":
        x_bounds = data.transform._machine_dims[0]
    if hasattr(data, "equilibrium"):
        data.transform.convert_to_rho_theta(t=data.t)

    fit, fit_err = run_gpr_fit(
        data,
        kernel_name=kernel_name,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        err_bounds=err_bounds,
        xdim=xdim,
    )

    if plot or save_fig:
        plt.ioff()
        fig_name = f"{pulse}_TS_Te"
        for tplot in data.t.values:
            plot_gpr_fit(
                data,
                fit,
                fit_err,
                tplot,
                ylabel="Te [eV]",
                xlabel="R [m]",
                title=str(st40.pulse),
                fig_name=f"{fig_name}_vs_R",
                save_fig=save_fig,
            )

            if hasattr(data, "equilibrium"):
                x_data = data.transform.rho.assign_coords(
                    R=("channel", data.R)
                ).swap_dims({"channel": "R"})
                x_fit = x_data.interp(R=fit.R)
                plot_gpr_fit(
                    data,
                    fit,
                    fit_err,
                    tplot,
                    x_data=x_data,
                    x_fit=x_fit,
                    ylabel="Te [eV]",
                    xlabel="rho-poloidal",
                    title=str(st40.pulse),
                    fig_name=f"{fig_name}_vs_rho",
                    save_fig=save_fig,
                )
            if not save_fig:
                plt.show()

    return data, fit, fit_err


if __name__ == "__main__":
    example_run()
