from copy import deepcopy
from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import xarray as xr
from xarray import DataArray

from indica.converters import LineOfSightTransform
from indica.numpy_typing import ArrayLike, LabeledArray
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_2d_map

DataArrayCoords = Tuple[DataArray, DataArray]


class InvertPoloidalAsymmetry:
    def __init__(
        self,
        knots: LabeledArray = [0, 0.2, 0.5, 0.9, 1.0],
        xspl: ArrayLike = np.linspace(0, 1.05, 51),
        bounds_profile: tuple = (0, np.inf),
        bounds_asymmetry: tuple = (-np.inf, np.inf),
        bc_profile: str = "clamped",
        bc_asymmetry: str = "natural",
    ):

        # xknots must include 0, 1, and rho_max
        rho_max = np.max(xspl)
        _xknots = np.append(np.array(knots), [0.0, 1.0, rho_max])
        _xknots = np.unique(np.sort(_xknots))
        xknots = _xknots[np.where((_xknots >= -0) * (_xknots <= rho_max))[0]]

        # Profile knots to scan <= 1.
        mask_profile = (xknots >= 0) * (xknots <= 1.0)
        indx_profile = np.where(mask_profile)[0]
        nknots_profile = len(indx_profile)

        # Asymmetry knots to scan >0 and <1.
        mask_asymmetry = (xknots > 0) * (xknots < 1.0)
        indx_asymmetry = np.where(mask_asymmetry)[0] + indx_profile[-1]
        nknots_asymmetry = len(indx_asymmetry)

        # Define optimisation bounds
        lbounds = np.append(
            [bounds_profile[0]] * nknots_profile,
            [bounds_asymmetry[0]] * nknots_asymmetry,
        )
        ubounds = np.append(
            [bounds_profile[1]] * nknots_profile,
            [bounds_asymmetry[1]] * nknots_asymmetry,
        )
        bounds = (lbounds, ubounds)

        self.xspl = xspl
        self.xknots = xknots
        self.nknots = np.size(xknots)
        self.bounds = bounds
        self.mask_profile = mask_profile
        self.mask_asymmetry = mask_asymmetry
        self.indx_profile = indx_profile
        self.indx_asymmetry = indx_asymmetry
        self.bc_profile = bc_profile
        self.bc_asymmetry = bc_asymmetry

    def __call__(
        self,
        los_integral: DataArray,
        los_transform: LineOfSightTransform,
        t: LabeledArray = None,
        debug: bool = False,
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

        coords = [("rho_poloidal", self.xspl)]
        mask_profile, mask_asymmetry = self.mask_profile, self.mask_asymmetry
        indx_profile, indx_asymmetry = self.indx_profile, self.indx_asymmetry
        xknots, xspl = self.xknots, self.xspl
        equilibrium = los_transform.equilibrium
        bc_profile, bc_asymmetry = self.bc_profile, self.bc_asymmetry

        def evaluate(xknots, yconcat):
            """
            1. Separate profile and asymmetry knots + add boundaries
            2. Initialize and call splines.
            3. Map profile and asymmetry to 2D
            """
            yprofile = yconcat[indx_profile]
            yprofile = np.append(yprofile, 0.0)
            yasymmetry = yconcat[indx_asymmetry]
            yasymmetry = np.append(yasymmetry[0], yasymmetry)
            yasymmetry = np.append(yasymmetry, [yasymmetry[-1], yasymmetry[-1]])

            profile_spline = CubicSpline(xknots, yprofile, 0, bc_profile,)
            profile_to_map = DataArray(profile_spline(xspl), coords=coords)
            asymmetry_spline = CubicSpline(xknots, yasymmetry, 0, bc_asymmetry,)
            asymmetry_parameter = DataArray(asymmetry_spline(xspl), coords=coords)
            profile_2d = centrifugal_asymmetry_2d_map(
                profile_to_map, asymmetry_parameter, equilibrium=equilibrium, t=t,
            )
            return profile_2d, profile_to_map, asymmetry_parameter

        def residuals(yknots_concat):
            """
            Calculate the residuals to minimize
            """
            profile_2d, _, _ = evaluate(xknots, yknots_concat)
            _bckc = los_transform.integrate_on_los(profile_2d, t=t)
            return (_data - _bckc) / _error

        if t is None:
            t = los_integral.t
        else:
            t = los_integral.t.sel(t=t, method="nearest")

        if hasattr(los_integral, "error"):
            error = los_integral.error
        else:
            error = los_integral * 0.1

        self.t = np.array(t, ndmin=1)
        self.los_transform = los_transform
        norm_factor = los_integral.max().values
        self.data = los_integral / norm_factor
        self.error = error / norm_factor

        # Initial guesses
        _guess_profile = self.data.sel(t=self.t[0]).mean().values
        _guess_asymmetry = 0.0
        guess_profile = np.full_like(xknots, _guess_profile)
        guess_asymmetry = np.full_like(xknots, _guess_asymmetry)
        yconcat = np.append(guess_profile[mask_profile], guess_asymmetry[mask_asymmetry],)

        profile = []
        asymmetry = []
        profile_2d = []
        bckc = []
        for t in self.t:
            _data = self.data.sel(t=t).values
            _error = self.error.sel(t=t).values
            _x = np.arange(len(_data))

            if debug:
                plt.figure()
                plt.ioff()
                plt.errorbar(_x, _data, _error, marker="o")

            fit = least_squares(residuals, yconcat, bounds=self.bounds, verbose=debug,)
            yconcat = fit.x

            _profile_2d, _profile_to_map, _asymmetry_parameter = evaluate(xknots, yconcat)
            _bckc = los_transform.integrate_on_los(_profile_2d, t=t)
            if debug:
                plt.plot(_x, _bckc, linewidth=2)
                plt.show()

            profile.append(_profile_to_map)
            asymmetry.append(_asymmetry_parameter)
            profile_2d.append(_profile_2d)
            bckc.append(_bckc)

        profile = xr.concat(profile, "t") * norm_factor
        asymmetry = xr.concat(asymmetry, "t")
        profile_2d = xr.concat(profile_2d, "t") * norm_factor
        bckc = xr.concat(bckc, "t") * norm_factor

        # xr.DataArray(
        #     np.empty((np.size(self.xspl), np.size(self.t))),
        #     coords=[("rho_poloidal", self.xspl), ("t", self.t)],
        # )

        return profile_2d, bckc, profile, asymmetry


def example_run():
    import indica.models.bolometer_camera as bolo
    import indica.operators.centrifugal_asymmetry as centrifugal_asymmetry
    from indica.operators.tomo_asymmetry_class import InvertPoloidalAsymmetry

    plasma, ion_density_2d = centrifugal_asymmetry.example_run()
    _, model, _ = bolo.example_run()

    profile_2d = ion_density_2d.sel(element="ar")
    los_transform = model.los_transform
    los_integral = los_transform.integrate_on_los(profile_2d, profile_2d.t.values)

    print("\n Running asymmetry inference...")
    self = InvertPoloidalAsymmetry()

    profile_2d_bckc, bckc, profile, asymmetry = self(
        los_integral, los_transform,
    )
    print("...and finished \n")

    for t in bckc.t:
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


if __name__ == "__main__":
    example_run()
