from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import xarray as xr
from xarray import DataArray

from indica.equilibrium import Equilibrium
from indica.numpy_typing import ArrayLike


def fit_profile_and_R_shift(
    Rdata: DataArray,
    zdata: DataArray,
    ydata: DataArray,
    yerr: DataArray,
    equilibrium: Equilibrium,
    xknots: ArrayLike = None,
    R_shift: DataArray = None,
    xspl: ArrayLike = np.linspace(0, 1.05, 51),
    bc_type: str = "clamped",
    bounds_R_shift: tuple = (-0.02, 0.02),
    verbose: bool = False,
):
    """Fit a profile and the R_shift of the equilibrium"""

    def residuals(all_knots):
        if R_shift is None:
            _R_shift = all_knots[-1]
            yknots = all_knots[:-1]
        else:
            _R_shift = R_shift.sel(t=t).values
            yknots = all_knots

        _xknots, _, _ = equilibrium.flux_coords(R + _R_shift, z, t=t)

        spline = CubicSpline(
            xknots,
            yknots,
            axis=0,
            bc_type=bc_type,
        )
        bckc = np.interp(_xknots, xspl, spline(xspl))

        _residuals = (y - bckc) / err

        return _residuals

    # Boundary conditions
    # values go to --> 0 outside separatrix (index = -2)
    nknots = np.size(xknots)
    lower_bound = np.full(nknots, -np.inf)
    upper_bound = np.full(nknots, np.inf)
    lower_bound[-1] = 0.0
    upper_bound[-1] = 0.01
    if R_shift is None:
        # If R_shift to be fitted, add additional knot and bounds
        nknots += 1
        lower_bound = np.append(lower_bound, bounds_R_shift[0])
        upper_bound = np.append(upper_bound, bounds_R_shift[1])

    # Initialize DataArray that will contain the final fit result
    yspl = xr.DataArray(
        np.empty((len(xspl), len(ydata.t))),
        coords=[("rho_poloidal", xspl), ("t", ydata.t.values)],
    )
    R_shift_fit = xr.DataArray(
        np.empty(len(ydata.t)),
        coords=[("t", ydata.t.values)],
    )
    rho_data = xr.full_like(ydata, np.nan)

    all_knots = None
    for t in ydata.t.values:
        # Normalize data so range of parameters to scan is all similar
        norm_factor = np.nanmax(ydata.sel(t=t).values)
        _y = ydata.sel(t=t).values / norm_factor
        _yerr = yerr.sel(t=t).values / norm_factor

        _R_shift = 0.0
        _yspl = np.full_like(xspl, 0.0)
        ind = np.where(np.isfinite(_y) * np.isfinite(_yerr))[0]
        if len(ind) > 2:
            R = Rdata[ind]
            z = zdata[ind]
            y = _y[ind]
            err = _yerr[ind]

            # Initial guess: profile linearly increasing edge>core & R_shift = 0.
            if all_knots is None:
                all_knots = np.linspace(np.max(y), 0, np.size(xknots))
                if R_shift is None:
                    all_knots = np.append(all_knots, 0.0)

            try:
                fit = least_squares(
                    residuals,
                    all_knots,
                    bounds=(lower_bound, upper_bound),
                    verbose=verbose,
                )

                if R_shift is None:
                    yknots = fit.x[:-1]
                    _R_shift = fit.x[-1]
                else:
                    yknots = fit.x
                    _R_shift = R_shift.sel(t=t)

                all_knots = deepcopy(fit.x)
                spline = CubicSpline(
                    xknots,
                    yknots,
                    axis=0,
                    bc_type=bc_type,
                )

                _yspl = spline(xspl) * norm_factor
            except ValueError:
                all_knots = None
                _R_shift = 0.0
                _yspl = np.full_like(xspl, 0.0)

        yspl.loc[dict(t=t)] = _yspl
        if R_shift is None:
            R_shift_fit.loc[dict(t=t)] = _R_shift

        _rho_data, _, _ = equilibrium.flux_coords(Rdata + _R_shift, zdata, t=t)
        rho_data.loc[dict(t=t)] = _rho_data

    if R_shift is not None:
        R_shift_fit = R_shift

    return yspl, R_shift_fit, rho_data
