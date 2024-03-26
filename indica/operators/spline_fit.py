import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import xarray as xr
from xarray import DataArray

from indica.numpy_typing import ArrayLike
from indica.readers.read_st40 import ReadST40


def fit_profile(
    xdata: DataArray,
    ydata: DataArray,
    yerr: DataArray,
    xspl: ArrayLike = np.linspace(0, 1.05, 51),
    knots: ArrayLike = None,
    virtual_knots: bool = False,
    bc_type: str = "clamped",
    verbose=0,
):
    """Fit a profile"""

    def residuals(yknots):
        spline = CubicSpline(
            xknots,
            yknots,
            axis=0,
            bc_type=bc_type,
        )
        bckc = np.interp(x, xspl, spline(xspl))
        return (y - bckc) / err

    yspl = xr.DataArray(
        np.empty((len(xspl), len(ydata.t))),
        coords=[("rho_poloidal", xspl), ("t", ydata.t.values)],
    )
    for t in ydata.t.values:
        _x = xdata.sel(t=t).values
        _y = ydata.sel(t=t).values
        _yerr = yerr.sel(t=t).values

        ind = np.where(np.isfinite(_x) * np.isfinite(_y) * np.isfinite(_yerr))[0]
        if len(ind) > 2:
            x_to_sort = _x[ind]
            y_to_sort = _y[ind]
            err_to_sort = _yerr[ind]

            ind_sort = np.argsort(x_to_sort)
            x = x_to_sort[ind_sort]
            y = y_to_sort[ind_sort]
            err = err_to_sort[ind_sort]

            if knots is None:
                xknots = np.append(np.append(0.0, x), 1.05)
            else:
                xknots = knots
            yknots = np.interp(xknots, x, y)
            yknots[-1] = 0.0

            if virtual_knots:
                ind = range(len(x) - 1)
                dx = [x[i + 1] - x[i] for i in ind]
                xknots_virt = [
                    x[i + 1] - dx[i] / 2 for i in ind if (i % 2 == 0 and x[i + 1] > 0.5)
                ]
                _xknots = np.unique(np.sort(np.append(xknots, xknots_virt)))
                yknots = np.interp(_xknots, xknots, yknots)
                xknots = _xknots

            lower_bound = np.full_like(xknots, -np.inf)
            upper_bound = np.full_like(xknots, np.inf)
            lower_bound[-1] = 0.0
            upper_bound[-1] = 0.01

            try:
                fit = least_squares(
                    residuals,
                    yknots,
                    bounds=(lower_bound, upper_bound),
                    verbose=verbose,
                )
                spline = CubicSpline(
                    xknots,
                    fit.x,
                    axis=0,
                    bc_type=bc_type,
                )
                _yspl = spline(xspl)
            except ValueError:
                _yspl = np.full_like(xspl, 0.0)
        else:
            _yspl = np.full_like(xspl, 0.0)

        yspl.loc[dict(t=t)] = _yspl

    return yspl


def example_run(
    pulse: int = 11314,
    tstart: float = 0.03,
    tend: float = 0.1,
    dt: float = 0.01,
    quantity: str = "te",
    R_shift: float = 0.0,
    knots: list = None,
    plot: bool = True,
):
    st40 = ReadST40(pulse, tstart=tstart, tend=tend, dt=dt)
    st40(["ts"], set_equilibrium=True, R_shift=R_shift)

    if quantity == "te" and knots is None:
        knots = [0, 0.3, 0.6, 0.8, 1.1]
    if quantity == "ne" and knots is None:
        knots = [0, 0.3, 0.6, 0.8, 0.95, 1.1]
    data_all = st40.raw_data["ts"][quantity]
    t = data_all.t
    transform = data_all.transform
    transform.convert_to_rho_theta(t=data_all.t)

    R = transform.R
    Rmag = transform.equilibrium.rmag.interp(t=t)

    # Fit all available TS data
    ind = np.full_like(data_all, True)
    rho = xr.where(ind, transform.rho, np.nan)
    data = xr.where(ind, data_all, np.nan)
    err = xr.where(ind, data_all.error, np.nan)
    fit = fit_profile(rho, data, err, knots=knots, virtual_knots=False)

    # Use only HFS channels
    ind = R <= Rmag
    rho_hfs = xr.where(ind, transform.rho, np.nan)
    data_hfs = xr.where(ind, data_all, np.nan)
    err_hfs = xr.where(ind, data_all.error, np.nan)
    fit_hfs = fit_profile(rho_hfs, data_hfs, err_hfs, knots=knots, virtual_knots=True)

    # Use only LFS channels
    ind = R >= Rmag
    rho_lfs = xr.where(ind, transform.rho, np.nan)
    data_lfs = xr.where(ind, data_all, np.nan)
    err_lfs = xr.where(ind, data_all.error, np.nan)
    fit_lfs = fit_profile(rho_lfs, data_lfs, err_lfs, knots=knots, virtual_knots=True)

    if plot:
        for t in data_all.t:
            plt.ioff()
            plt.errorbar(
                rho_hfs.sel(t=t),
                data_hfs.sel(t=t),
                err_hfs.sel(t=t),
                marker="o",
                label="data HFS",
                color="blue",
            )
            plt.errorbar(
                rho_lfs.sel(t=t),
                data_lfs.sel(t=t),
                err_lfs.sel(t=t),
                marker="o",
                label="data LFS",
                color="red",
            )
            fit.sel(t=t).plot(
                linewidth=5, alpha=0.5, color="black", label="spline fit all"
            )
            fit_lfs.sel(t=t).plot(
                linewidth=5, alpha=0.5, color="red", label="spline fit LFS"
            )
            fit_hfs.sel(t=t).plot(
                linewidth=5, alpha=0.5, color="blue", label="spline fit HFS"
            )
            plt.legend()
            plt.show()

    return data_all, fit


if __name__ == "__main__":
    plt.ioff()
    example_run(11089, quantity="ne")
    plt.show()
