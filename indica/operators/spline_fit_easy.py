from indica.numpy_typing import ArrayLike
import numpy as np
import xarray as xr
from xarray import DataArray

from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import least_squares
import matplotlib.pylab as plt

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
        spline = CubicSpline(xknots, yknots, axis=0, bc_type=bc_type,)
        bckc = np.interp(x, xspl, spline(xspl))
        residuals = np.sqrt((y - bckc) ** 2 / err ** 2)
        return residuals

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
                spline = CubicSpline(xknots, fit.x, axis=0, bc_type=bc_type,)
                _yspl = spline(xspl)
            except ValueError:
                _yspl = np.full_like(xspl, 0.0)
        else:
            _yspl = np.full_like(xspl, 0.0)

        yspl.loc[dict(t=t)] = _yspl

    return yspl


def spline_fit_ts(
    pulse: int, tstart: float = 0.0, tend: float = 0.2, plot: bool = False,
):

    ST40 = ReadST40(pulse, tstart=tstart, tend=tend)
    ST40(["ts"])

    Te_data_all = ST40.binned_data["ts"]["te"]
    t = ST40.binned_data["ts"]["te"].t
    transform = ST40.binned_data["ts"]["te"].transform
    R = transform.R
    Rmag = transform.equilibrium.rmag.interp(t=t)

    # Fit all available TS data
    ind = np.full_like(Te_data_all, True)
    rho = xr.where(ind, transform.rho, np.nan)
    Te_data = xr.where(ind, Te_data_all, np.nan)
    Te_err = xr.where(ind, Te_data_all.error, np.nan)
    Te_fit = fit_profile(
        rho, Te_data, Te_err, knots=[0, 0.3, 0.5, 0.75, 0.95, 1.05], virtual_knots=False
    )

    # Use only HFS channels
    ind = R <= Rmag
    rho = xr.where(ind, transform.rho, np.nan)
    Te_data = xr.where(ind, Te_data_all, np.nan)
    Te_err = xr.where(ind, Te_data_all.error, np.nan)
    Te_fit_hfs = fit_profile(rho, Te_data, Te_err, virtual_knots=True)

    if plot:
        for t in Te_data_all.t:
            plt.ioff()
            plt.errorbar(
                Te_data_all.transform.rho.sel(t=t),
                Te_data_all.sel(t=t),
                Te_data_all.error.sel(t=t),
                marker="o",
                label="data",
            )
            Te_fit.sel(t=t).plot(
                linewidth=5, alpha=0.5, color="orange", label="spline fit all"
            )
            Te_fit_hfs.sel(t=t).plot(
                linewidth=5, alpha=0.5, color="red", label="spline fit HFS"
            )
            plt.legend()
            plt.show()

    return Te_data_all, Te_fit


if __name__ == "__main__":
    spline_fit_ts(
        10619, tstart=0.0, tend=0.2, plot=True,
    )
