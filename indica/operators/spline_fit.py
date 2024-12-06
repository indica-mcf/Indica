import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import xarray as xr
from xarray import DataArray

from indica.numpy_typing import ArrayLike


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
        coords=[("rhop", xspl), ("t", ydata.t.values)],
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
