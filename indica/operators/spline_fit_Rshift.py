from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import xarray as xr
from xarray import DataArray

from indica.equilibrium import Equilibrium
from indica.numpy_typing import ArrayLike
from indica.readers.read_st40 import ReadST40


def fit_profile_and_Rshift(
    Rdata: DataArray,
    zdata: DataArray,
    ydata: DataArray,
    yerr: DataArray,
    equilibrium: Equilibrium,
    xknots: ArrayLike = None,
    Rshift: DataArray = None,
    xspl: ArrayLike = np.linspace(0, 1.05, 51),
    bc_type: str = "clamped",
    bounds_Rshift: tuple = (-0.02, 0.02),
    verbose: bool = False,
):
    """Fit a profile and the R_shift of the equilibrium"""

    def residuals(all_knots):
        if Rshift is None:
            _Rshift = all_knots[-1]
            yknots = all_knots[:-1]
        else:
            _Rshift = Rshift.sel(t=t).values
            yknots = all_knots

        _xknots, _, _ = equilibrium.flux_coords(R + _Rshift, z, t=t)

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
    if Rshift is None:
        # If Rshift to be fitted, add additional knot and bounds
        nknots += 1
        lower_bound = np.append(lower_bound, bounds_Rshift[0])
        upper_bound = np.append(upper_bound, bounds_Rshift[1])

    # Initialize DataArray that will contain the final fit result
    yspl = xr.DataArray(
        np.empty((len(xspl), len(ydata.t))),
        coords=[("rho_poloidal", xspl), ("t", ydata.t.values)],
    )
    Rshift_fit = xr.DataArray(
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

        _Rshift = 0.0
        _yspl = np.full_like(xspl, 0.0)
        ind = np.where(np.isfinite(_y) * np.isfinite(_yerr))[0]
        if len(ind) > 2:
            R = Rdata[ind]
            z = zdata[ind]
            y = _y[ind]
            err = _yerr[ind]

            # Initial guess: profile linearly increasing edge>core & Rshift = 0.
            if all_knots is None:
                all_knots = np.linspace(np.max(y), 0, np.size(xknots))
                if Rshift is None:
                    all_knots = np.append(all_knots, 0.0)

            try:
                fit = least_squares(
                    residuals,
                    all_knots,
                    bounds=(lower_bound, upper_bound),
                    verbose=verbose,
                )

                if Rshift is None:
                    yknots = fit.x[:-1]
                    _Rshift = fit.x[-1]
                else:
                    yknots = fit.x
                    _Rshift = Rshift.sel(t=t)

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
                _Rshift = 0.0
                _yspl = np.full_like(xspl, 0.0)

        yspl.loc[dict(t=t)] = _yspl
        if Rshift is None:
            Rshift_fit.loc[dict(t=t)] = _Rshift

        _rho_data, _, _ = equilibrium.flux_coords(Rdata + _Rshift, zdata, t=t)
        rho_data.loc[dict(t=t)] = _rho_data

    if Rshift is not None:
        Rshift_fit = Rshift

    return yspl, Rshift_fit, rho_data


def example_run(
    pulse: int = 11314,
    tstart: float = 0.03,
    tend: float = 0.13,
    dt: float = 0.01,
    quantity: str = "te",
    xknots: list = None,
    verbose: bool = False,
):
    st40 = ReadST40(pulse, tstart=tstart, tend=tend, dt=dt)
    st40(["ts"])

    if xknots is None:
        if quantity == "te":
            xknots = [0, 0.4, 0.6, 0.8, 1.1]
        elif quantity == "ne":
            xknots = [0, 0.4, 0.8, 0.95, 1.1]
        else:
            raise ValueError

    data = st40.raw_data["ts"][quantity]
    err = data.error
    transform = data.transform
    equilibrium = transform.equilibrium
    R = transform.R
    z = transform.z

    fit, Rshift, rho_data = fit_profile_and_Rshift(
        R, z, data, err, equilibrium, xknots=xknots, verbose=verbose
    )

    for t in data.t:
        _Rshift = Rshift.sel(t=t).values
        rho = rho_data.sel(t=t)

        plt.ioff()
        plt.errorbar(
            rho,
            data.sel(t=t),
            err.sel(t=t),
            marker="o",
            label="data",
            color="blue",
        )
        fit.sel(t=t).plot(linewidth=5, alpha=0.5, color="black", label="spline fit all")
        plt.title(
            f"pulse={pulse}, t={int(t*1000.)} ms, R_shift={(_Rshift*100.):.1f} cm"
        )
        plt.legend()
        plt.show()

    return data, fit


if __name__ == "__main__":
    plt.ioff()
    example_run(11089, quantity="ne")
    plt.show()
