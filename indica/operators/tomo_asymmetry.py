from copy import deepcopy
from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import xarray as xr
from xarray import DataArray

from indica.converters import LineOfSightTransform
from indica.numpy_typing import ArrayLike
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_2d_map

DataArrayCoords = Tuple[DataArray, DataArray]


def InvertPoloidalAsymmetry(
    los_integral: DataArray,
    los_transform: LineOfSightTransform,
    knots: list = [0, 0.2, 0.5, 0.9, 1.0],
    xspl: ArrayLike = np.linspace(0, 1.05, 51),
    bc_profile: str = "clamped",
    bc_asymmetry: str = "natural",
    times: ArrayLike = None,
):
    """
    Estimates the poloidal distribution from line-of-sight integrals
    assuming the local quantity is poloidally asymmetric following
    Wesson's formula.

    Parameters
    ----------
    los_integral
        Measured LOS-integral from e.g. a pinhole camera.
    los_transform
        Line of sight transform for the forward model (must already have
        assigned equilibrium)
    knots
        The spline knots to use when fitting the emissivity data.
    """

    def residuals(yknots_concat):
        yknots_profile = yknots_concat[:nknots]
        yknots_asymmetry = yknots_concat[nknots:]

        profile_spline = CubicSpline(
            xknots,
            yknots_profile,
            axis=0,
            bc_type=bc_profile,
        )
        asymmetry_spline = CubicSpline(
            xknots,
            yknots_asymmetry,
            axis=0,
            bc_type=bc_asymmetry,
        )

        profile_to_map = DataArray(profile_spline(xspl), coords=coords)
        asymmetry_parameter = DataArray(asymmetry_spline(xspl), coords=coords)

        profile_2d = centrifugal_asymmetry_2d_map(
            profile_to_map,
            asymmetry_parameter,
            equilibrium=los_transform.equilibrium,
            t=t,
        )
        _bckc = los_transform.integrate_on_los(profile_2d, t=t)
        return (_data - _bckc) / _error

    if times is None:
        times = los_integral.t

    if hasattr(los_integral, "error"):
        error = los_integral.error
    else:
        error = los_integral * 0.1

    coords = [("rho_poloidal", xspl)]

    nknots = np.size(knots)
    xknots = np.array(knots)
    # xknots_concat = np.append(np.array(knots), np.array(knots))

    guess = los_integral.isel(t=0).mean().values
    yknots_emission = np.full_like(xknots, guess)
    yknots_asymmetry = np.full_like(xknots, 0)
    yknots_concat = np.append(yknots_emission, yknots_asymmetry)

    lower_bound, upper_bound = set_bounds(xknots)

    profile = xr.DataArray(
        np.empty((np.size(xspl), np.size(times))),
        coords=[("rho_poloidal", xspl), ("t", times)],
    )
    asymmetry = deepcopy(profile)
    profile_2d = []
    bckc = xr.full_like(los_integral, np.nan)
    for t in times:
        _data = los_integral.sel(t=t)
        _error = error.sel(t=t)

        fit = least_squares(
            residuals,
            yknots_concat,
            bounds=(lower_bound, upper_bound),
            verbose=True,
        )

        profile_spline = CubicSpline(
            xknots,
            fit.x[:nknots],
            axis=0,
            bc_type=bc_profile,
        )
        asymmetry_spline = CubicSpline(
            xknots,
            fit.x[nknots:],
            axis=0,
            bc_type=bc_asymmetry,
        )

        profile.loc[dict(t=t)] = profile_spline(xspl)
        asymmetry.loc[dict(t=t)] = asymmetry_spline(xspl)

        _profile_2d = centrifugal_asymmetry_2d_map(
            profile,
            asymmetry,
            equilibrium=los_transform.equilibrium,
            t=t,
        )
        bckc.loc[dict(t=t)] = los_transform.integrate_on_los(_profile_2d, t=t)
        profile_2d.append(_profile_2d)

    profile_2d = xr.concat(profile_2d, "t")

    return profile_2d, bckc, profile, asymmetry


def set_bounds(xknots):
    l_bound_profile = np.full_like(xknots, 0)
    u_bound_profile = np.full_like(xknots, 1)
    l_bound_asymmetry = np.full_like(xknots, -10)
    u_bound_asymmetry = np.full_like(xknots, 10)
    l_bound_asymmetry[0] = 0
    u_bound_asymmetry[0] = 1

    lower_bound = np.append(l_bound_profile, l_bound_asymmetry)
    upper_bound = np.append(u_bound_profile, u_bound_asymmetry)

    return lower_bound, upper_bound


def example_run():
    import indica.models.bolometer_camera as bolo
    import indica.operators.centrifugal_asymmetry as centrifugal_asymmetry

    plasma, ion_density_2d = centrifugal_asymmetry.example_run()
    _, model, _ = bolo.example_run()

    profile_2d = ion_density_2d.sel(element="ar")
    los_transform = model.los_transform
    los_integral = los_transform.integrate_on_los(profile_2d, profile_2d.t.values)

    times = los_integral.t.values[:4]
    norm_factor = los_integral.max().values
    profile_2d_bckc, bckc, profile_bckc, asymmetry_bckc = InvertPoloidalAsymmetry(
        los_integral / norm_factor, los_transform, times=times
    )
    profile_2d_bckc, bckc = profile_2d_bckc * norm_factor, bckc * norm_factor

    for t in times:
        plt.ioff()
        plt.figure()
        los_integral.sel(t=t).plot(marker="o")
        bckc.sel(t=t).plot()
        plt.figure()
        ((profile_2d.sel(t=t) - profile_2d_bckc.sel(t=t)) / profile_2d.sel(t=t)).plot()
        plt.figure()
        profile_2d.sel(t=t).sel(z=0, method="nearest").plot(marker="o")
        profile_2d_bckc.sel(t=t).sel(z=0, method="nearest").plot()
        plt.show()
