"""Inverts soft X-ray data to estimate the emissivity of the plasma."""

from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
import warnings

import numpy as np
from scipy.integrate import romb
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from xarray import concat
from xarray import DataArray
from xarray import where

from .abstractoperator import Operator
from ..converters import bin_to_time_labels
from ..converters import CoordinateTransform
from ..converters import FluxMajorRadCoordinates
from ..converters import FluxSurfaceCoordinates
from ..converters import ImpactParameterCoordinates
from ..converters import LinesOfSightTransform
from ..datatypes import DataType
from ..numpy_typing import ArrayLike
from ..session import global_session
from ..session import Session

DataArrayCoords = Tuple[DataArray, DataArray, DataArray]


class EmissivityProfile:
    """Callable class calculating an estimated emissivity profile for the
    plasma.

    Parameters
    ----------
    symmetric_emissivity : DataArray
        Estimate of the profile of the symmetric emissivity at each knot in rho.
    asymmetry_parameter : DataArray
        Parameter describing asymmetry in the emissivity between high and low
        flux surfaces at each knot in rho.
    coord_transform : FluxSurfaceCoordinates
        The flux coordinate system which should be used.
    time : DataArray
        The times this profile is for. If not present will try to use time axis
        from `symmetric_emissivity` or `asymmetry_parameter`.

    """

    def __init__(
        self,
        symmetric_emissivity: ArrayLike,
        asymmetry_parameter: ArrayLike,
        coord_transform: FluxSurfaceCoordinates,
        time: Optional[DataArray] = None,
    ):
        self._dim = "rho_" + coord_transform.flux_kind
        sym_axis = symmetric_emissivity.dims.index(self._dim)
        self._interp_sym_coords = (
            lambda new_coords: [
                (d, symmetric_emissivity.coords[d])
                for d in symmetric_emissivity.dims[0:sym_axis]
            ]
            + new_coords
            + [
                (d, symmetric_emissivity.coords[d])
                for d in symmetric_emissivity.dims[sym_axis + 1 :]
            ]
        )
        self.symmetric_emissivity = CubicSpline(
            symmetric_emissivity.coords[self._dim],
            symmetric_emissivity,
            sym_axis,
            (
                (2, np.zeros_like(symmetric_emissivity.isel({self._dim: 0}))),
                (2, np.zeros_like(symmetric_emissivity.isel({self._dim: -1}))),
            ),
            False,
        )
        asym_axis = asymmetry_parameter.dims.index(self._dim)
        self._interp_asym_coords = (
            lambda new_coords: [
                (d, asymmetry_parameter.coords[d])
                for d in asymmetry_parameter.dims[0:asym_axis]
            ]
            + new_coords
            + [
                (d, asymmetry_parameter.coords[d])
                for d in asymmetry_parameter.dims[asym_axis + 1 :]
            ]
        )
        self.asymmetry_parameter = CubicSpline(
            asymmetry_parameter.coords[self._dim],
            asymmetry_parameter,
            asym_axis,
            (
                (2, np.zeros_like(asymmetry_parameter.isel({self._dim: 0}))),
                (2, np.zeros_like(asymmetry_parameter.isel({self._dim: -1}))),
            ),
        )
        self.transform = FluxMajorRadCoordinates(coord_transform)
        self.rho = "rho_" + self.transform.flux_kind
        self.emissivity_rho_95 = self.symmetric_emissivity(0.95)
        self.beta = 40.0
        self.coefficient = self.emissivity_rho_95 / (np.exp(-self.beta * 0.05) - 1.0)
        if time is None:
            time_sym = symmetric_emissivity.coords.get("t", None)
            time_asym = asymmetry_parameter.coords.get("t", None)
            if time_sym is None and time_asym is None:
                raise ValueError(
                    "Neither `symmetric_emissivity` and `asymmetry_parameter` "
                    "have dimension 't' and argument `time` not provided"
                )
            elif not time_sym.equals(time_asym):
                raise ValueError(
                    "Dimension 't' must match for `symmetric_emissivity` and "
                    "`asymmetry_parameter`"
                )
            self.time = time_sym if time_sym is not None else time_asym
        else:
            self.time = time

    def __call__(
        self,
        coord_system: CoordinateTransform,
        x1: DataArray,
        x2: DataArray,
        t: DataArray,
    ) -> DataArray:
        rho, R, t = cast(
            DataArrayCoords, coord_system.convert_to(self.transform, x1, x2, t)
        )
        return self.evaluate(rho, R, t)

    def evaluate(
        self,
        rho: DataArray,
        R: DataArray,
        t: Optional[DataArray] = None,
        R_0: Optional[DataArray] = None,
    ) -> DataArray:
        """Evaluate the function at a location defined using (R, z) coordinates"""
        # print([(d, rho.coords[d]) for d in rho.dims if d != self._dim])
        # print(self._interp_sym_coords(
        #         [(d, rho.coords[d]) for d in rho.dims if d != self._dim]
        #     )
        # )
        symmetric = DataArray(
            self.symmetric_emissivity(rho),
            coords=self._interp_sym_coords([(d, rho.coords[d]) for d in rho.dims]),
        )
        asymmetric = DataArray(
            self.asymmetry_parameter(rho),
            coords=self._interp_asym_coords([(d, rho.coords[d]) for d in rho.dims]),
        )
        if t is None:
            t = self.time
            if t in symmetric.dims:
                symmetric = symmetric.interp(t=t, method="cubic")
            if t in asymmetric.dims:
                asymmetric = asymmetric.interp(t=t, method="cubic")
        if R_0 is None:
            R_0 = cast(DataArray, self.transform.equilibrium.R_hfs(rho, t)[0])
        result = symmetric * np.exp(asymmetric * (R ** 2 - R_0 ** 2))
        # Ensure round-off error doesn't result in any values below 0
        return where(result < 0.0, 0.0, result).fillna(0.0)


def integrate_los(
    los: LinesOfSightTransform,
    x1: DataArray,
    x2: DataArray,
    t: DataArray,
    emissivity: EmissivityProfile,
    n: int = 65,
) -> DataArray:

    """Integrate the emissivity profile along the line of sight for the
    given time(s).

    Parameters
    ----------
    los
        The line of sight coordinate system along which to integrate.
    emissivity
        The emissivity profile to be integrated.
    n
        The (minimum) number of samples with which to integrate. Actual
        number will be the smallest value ``2**k + 1 >= n``.
    """
    n = 2 ** int(np.ceil(np.log(n - 1) / np.log(2))) + 1
    x2 = DataArray(np.linspace(0.0, 1.0, n), dims="x2")
    distances = los.distance("x2", DataArray(0), x2[0:2], emissivity.time)
    dl = cast(DataArray, distances)[1]
    emissivity_vals = emissivity(los, x1, x2, t)
    axis = emissivity_vals.dims.index("x2")
    return romb(emissivity_vals, dl, axis)


class InvertSXR(Operator):
    """Estimates the emissivity distribution of the plasma using soft X-ray
    data.

    Parameters
    ----------
    flux_coordinates : FluxSurfaceCoordinates
        The flux surface coordinate system on which to calculate the fit.
    rho : DataArray
        Flux surface coordinates on which to return emissivity result.
    theta : DataArray
        Theta coordinates on which to return emissivity result.
    t : DataArray
        Time coordinatse on which to return emissivity result.
    num_cameras : int
        The number of cameras to which the data will be fit.
    n_knots : int
        The number of  spline knots to use when fitting the emissivity data.
    sess : Session
        An object representing the session being run. Contains information
        such as provenance data.
    rho_grid : DataArray
        The flux surface coordinates on which to return emissivity data.
    theta_grid : DataArray
        The poloidal angle coordinates on which to return emissivity data.
    time_grid : DataArray
        The times at which to return emissivity data.

    """

    RETURN_TYPES: List[DataType] = [("emissivity", "sxr")]

    def __init__(
        self,
        flux_coordinates: FluxSurfaceCoordinates,
        rho: DataArray,
        theta: DataArray,
        t: DataArray,
        num_cameras: int = 1,
        n_knots: int = 6,
        sess: Session = global_session,
    ):
        self.n_knots = n_knots
        self.flux_coords = flux_coordinates
        self.rho = rho
        self.theta = theta
        self.t = t
        self.num_cameras = num_cameras
        self.ARGUMENT_TYPES: List[DataType] = [("luminous_flux", "sxr")] * num_cameras
        super().__init__(
            sess,
            flux_coordinates=flux_coordinates.encode(),
            num_cameras=num_cameras,
            n_knots=n_knots,
        )

    def __call__(self, *cameras: DataArray) -> DataArray:  # type: ignore[override]
        """Calculate the emissivity profile for the plasma.

        Parameters
        ----------
        cameras
            The luminosity data being fit to, with each camera passed
            as a separate argument.

        Returns
        -------
        :
            Emissivity values on the (\rho, \theta) grid specified when
            this operator was constructed.
        """
        self.validate_arguments(*cameras)
        n = self.n_knots
        num_los = [len(c.attrs["transform"].default_x1) for c in cameras]
        print("Calculating impact parameters")
        impact_params = [
            ImpactParameterCoordinates(c.attrs["transform"], self.flux_coords)
            for c in cameras
        ]
        binned_cameras = [bin_to_time_labels(self.t.data, c) for c in cameras]
        rho_max = max(ip.rhomax() for ip in impact_params)
        knots = np.empty(n)
        knots[0 : n - 1] = np.linspace(0, 1.0, n - 1) ** 1.2 * float(rho_max)
        knots[-1] = 1.0
        dim_name = "rho_" + self.flux_coords.flux_kind

        def knotvals_to_xarray(knotvals):
            symmetric_emissivity = DataArray(np.empty(n), coords=[(dim_name, knots)])
            symmetric_emissivity[0:-1] = knotvals[0 : n - 1]
            symmetric_emissivity[-1] = 0.0
            asymmetry_parameter = DataArray(np.empty(n), coords=[(dim_name, knots)])
            asymmetry_parameter[0] = 0.5 * knotvals[n - 1]
            asymmetry_parameter[1:-1] = knotvals[n - 1 :]
            asymmetry_parameter[-1] = 0.5 * knotvals[-1]
            return symmetric_emissivity, asymmetry_parameter

        def residuals(
            knotvals: np.ndarray,
            rho: List[DataArray],
            R: List[DataArray],
            time: DataArray,
            R_0: List[DataArray],
            dl: List[float],
        ) -> np.ndarray:
            symmetric_emissivity, asymmetry_parameter = knotvals_to_xarray(knotvals)
            estimate = EmissivityProfile(
                symmetric_emissivity, asymmetry_parameter, self.flux_coords, time
            )
            start = 0
            resid = np.empty(sum(num_los))
            for camera, n_los, ip_coords, rho_cam, Rcam, R0, dlcam in zip(
                binned_cameras, num_los, impact_params, rho, R, R_0, dl
            ):
                end = start + n_los
                emissivity_vals = estimate.evaluate(rho_cam, Rcam, R_0=R0)
                axis = emissivity_vals.dims.index("x2")
                integral = romb(emissivity_vals, dlcam, axis)
                resid[start:end] = (
                    (camera.sel(t=time) - integral)
                    / (
                        camera.sel(t=time)
                        * (
                            0.02
                            + 0.18
                            * np.abs(
                                ip_coords.rho_min.sel(t=time).rename(
                                    index=camera.attrs["transform"].x1_name
                                )
                            )
                        )
                    )
                ).data
                start = end
            assert np.all(np.isfinite(resid))
            return resid

        x2 = DataArray(np.linspace(0.0, 1.0, 65), dims="x2")
        dls = [
            cast(
                DataArray,
                c.attrs["transform"].distance("x2", DataArray(0), x2[0:2], 0.0)[0],
            )[1]
            for c in cameras
        ]

        symmetric_emissivities: List[DataArray] = []
        asymmetry_parameters: List[DataArray] = []
        abel_inversion = np.linspace(3e3, 0.0, n - 1)
        guess = np.concatenate((abel_inversion, np.zeros(n - 2)))
        bounds = (
            np.concatenate((np.zeros(n - 1), -0.5 * np.ones(n - 2))),
            np.concatenate((1e6 * np.ones(n - 1), np.ones(n - 2))),
        )
        rho_maj_rad = FluxMajorRadCoordinates(self.flux_coords)

        for t in np.asarray(self.t):
            print(f"Solving for t={t}")
            print("-----------------\n")
            rhos, Rs, _ = zip(
                *[
                    c.attrs["transform"].convert_to(rho_maj_rad, x2=x2, t=t)
                    for c in cameras
                ]
            )
            R_0s = [
                cast(DataArray, ip_coords.equilibrium.R_hfs(rho, t)[0])
                for ip_coords, rho in zip(impact_params, rhos)
            ]
            fit = least_squares(
                residuals,
                guess,
                bounds=bounds,
                args=(rhos, Rs, t, R_0s, dls),
                verbose=2,
            )
            if fit.status == -1:
                raise RuntimeError(
                    "Improper input to `least_squares` function when trying to "
                    "fit emissivity to SXR data."
                )
            elif fit.status == 0:
                warnings.warn(
                    f"Attempt to fit emissivity to SXR data at time t={t} "
                    "reached maximum number of function evaluations.",
                    RuntimeWarning,
                )
            sym, asym = knotvals_to_xarray(fit.x)
            symmetric_emissivities.append(sym)
            asymmetry_parameters.append(asym)
            guess = fit.x
        symmetric_emissivity = concat(symmetric_emissivities, dim=self.t)
        asymmetry_parameter = concat(asymmetry_parameters, dim=self.t)
        estimate = EmissivityProfile(
            symmetric_emissivity, asymmetry_parameter, self.flux_coords
        )
        result = estimate(self.flux_coords, self.rho, self.theta, self.t)
        result.attrs["transform"] = self.flux_coords
        result.attrs["datatype"] = ("emissivity", "sxr")
        result.attrs["provenance"] = self.create_provenance()
        result.attrs["emissivity"] = estimate
        result.name = "sxr_emissivity"
        return result
