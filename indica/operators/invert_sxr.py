"""Inverts soft X-ray data to estimate the emissivity of the plasma."""

from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

import numpy as np
from scipy.integrate import romb
from scipy.interpolate import interp1d
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
    knots : ArrayLike
        Flux surface coordinates at which values of values of the other
        arguments are known.
    symmetric_emissivity : DataArray
        Estimate of the profile of the symmetric emissivity at each knot.
    asymmetry_parameter : DataArray
        Parameter describing asymmetry in the emissivity between high and low
        flux surfaces at each knot.
    coord_transform : FluxSurfaceCoordinates
        The flux coordinate system which should be used.

    """

    def __init__(
        self,
        knots: ArrayLike,
        symmetric_emissivity: ArrayLike,
        asymmetry_parameter: ArrayLike,
        coord_transform: FluxSurfaceCoordinates,
        time: DataArray,
    ):
        self.symmetric_emissivity = interp1d(
            knots, symmetric_emissivity, fill_value="extrapolate"
        )
        self.asymmetry_parameter = interp1d(
            knots, asymmetry_parameter, fill_value="extrapolate"
        )
        self.transform = FluxMajorRadCoordinates(coord_transform)
        self.rho = "rho_" + self.transform.flux_kind
        self.emissivity_rho_95 = self.symmetric_emissivity(0.95)
        self.beta = 40.0
        self.coefficient = self.emissivity_rho_95 / (np.exp(-self.beta * 0.05) - 1.0)
        self.time = time

    def __call__(
        self,
        coord_system: CoordinateTransform,
        x1: Optional[DataArray] = None,
        x2: Optional[DataArray] = None,
    ) -> DataArray:
        rho, R, _ = cast(
            DataArrayCoords, coord_system.convert_to(self.transform, x1, x2, self.time)
        )
        return self.evaluate(rho, R)

    def evaluate(
        self, rho: DataArray, R: DataArray, R_0: Optional[DataArray] = None
    ) -> DataArray:
        """Evaluate the function at a location defined using (R, z) coordinates
        """
        if R_0 is None:
            R_0 = cast(DataArray, self.transform.equilibrium.R_hfs(rho, self.time)[0])
        symmetric = self.symmetric_emissivity(rho)
        mask = np.logical_and(rho > 0.95, rho < 1.0)
        if np.any(mask):
            symmetric = where(
                mask,
                self.emissivity_rho_95
                + self.coefficient * (1 - np.exp(-self.beta * (rho - 0.95))),
                symmetric,
            )
        mask = rho >= 1.0
        if np.any(mask):
            symmetric = where(mask, 0.0, symmetric)
        asymmetric = self.asymmetry_parameter(rho)
        return symmetric * np.exp(asymmetric * (R ** 2 - R_0 ** 2))


def integrate_los(
    los: LinesOfSightTransform, emissivity: EmissivityProfile, n: int = 65
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
    distances, _ = los.distance("x2", DataArray(0), x2[0:2], emissivity.time)
    dl = cast(DataArray, distances)[1]
    assert isinstance(los.default_x1, DataArray)
    emissivity_vals = emissivity(los, los.default_x1, x2)
    axis = emissivity_vals.dims.index("x2")
    return romb(emissivity_vals, dl, axis)


class InvertSXR(Operator):
    """Estimates the emissivity distribution of the plasma using soft X-ray
    data.

    Parameters
    ----------
    delta_rho_factor : Union[int, float]
        The number of times the spacing of spline knots should be grather than
        the spacing of the impact parameters of lines of sight.
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
        num_cameras: int = 1,
        delta_rho_factor: Union[int, float] = 4,
        sess: Session = global_session,
    ):
        self.delta_rho_factor = delta_rho_factor
        self.flux_coords = flux_coordinates
        assert isinstance(flux_coordinates.default_t, DataArray)
        self.t = flux_coordinates.default_t
        self.num_cameras = num_cameras
        self.ARGUMENT_TYPES: List[DataType] = [("luminous_flux", "sxr")] * num_cameras
        self.estimate: EmissivityProfile
        super().__init__(
            sess,
            flux_coordinates=flux_coordinates.encode(),
            num_cameras=num_cameras,
            delta_rho_factor=delta_rho_factor,
        )

    def __call__(  # type: ignore[override]
        self, *cameras: DataArray
    ) -> Tuple[DataArray, DataArray]:
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
        num_los = [len(c.attrs["transform"].default_x1) for c in cameras]
        print("Calculating impact parameters")
        impact_params = [
            ImpactParameterCoordinates(c.attrs["transform"], self.flux_coords)
            for c in cameras
        ]
        binned_cameras = [bin_to_time_labels(self.t.data, c) for c in cameras]
        total_los = np.sum(num_los)
        average_knot_spacing = (
            self.delta_rho_factor
            * np.sum([ip.drho() * n for ip, n in zip(impact_params, num_los)])
            / total_los
        )
        # Do I use different ones at each timestep or what?
        rho_max = max(ip.rhomax() for ip in impact_params)
        n = int((rho_max / average_knot_spacing).round()) + 2
        average_knot_spacing = float(rho_max) / (n - 2)
        spacing = average_knot_spacing * np.linspace(1.2, 0.8, n - 1)
        knots = np.empty(n)
        knots[0] = 0.0
        knots[1:] = np.cumsum(spacing)

        def residuals(
            knotvals: np.ndarray,
            rho: List[DataArray],
            R: List[DataArray],
            time: DataArray,
            R_0: List[DataArray],
            dl: List[float],
        ) -> np.ndarray:
            symmetric_emissivity = knotvals[:n]
            asymmetry_parameter = np.empty(n)
            asymmetry_parameter[0] = 0.5 * knotvals[n]
            asymmetry_parameter[1:-1] = knotvals[n:]
            asymmetry_parameter[-1] = 0.5 * knotvals[-1]
            self.estimate = EmissivityProfile(
                knots, symmetric_emissivity, asymmetry_parameter, self.flux_coords, time
            )
            start = 0
            resid = np.empty(sum(num_los))
            for camera, n_los, ip_coords, rho_cam, Rcam, R0, dlcam in zip(
                binned_cameras, num_los, impact_params, rho, R, R_0, dl
            ):
                end = start + n_los
                emissivity_vals = self.estimate.evaluate(rho_cam, Rcam, R0)
                axis = emissivity_vals.dims.index("x2")
                integral = romb(emissivity_vals, dlcam, axis)
                resid[start:end] = (
                    camera.sel(t=time)
                    - integral
                    / (
                        camera.attrs["error"].sel(t=time)
                        * (
                            0.8
                            + 0.4
                            * ip_coords.rho_min.sel(t=time).rename(
                                index=camera.attrs["x1"]
                            )
                        )
                    )
                ).data
                start = end
            return resid

        x2 = DataArray(np.linspace(0.0, 1.0, 65), dims="x2")
        dls = [
            cast(
                DataArray,
                c.attrs["transform"].distance("x2", DataArray(0), x2[0:2], 0.0)[0],
            )[1]
            for c in cameras
        ]

        results: List[DataArray] = []
        abel_inversion = np.zeros(n)
        guess = np.concatenate((abel_inversion, np.zeros(n - 2)))
        rho_maj_rad = FluxMajorRadCoordinates(self.flux_coords)

        # FOR DEBUG PURPOSES
        knotvals = guess
        symmetric_emissivity = knotvals[:n]
        asymmetry_parameter = np.empty(n)
        asymmetry_parameter[0] = 0.5 * knotvals[n]
        asymmetry_parameter[1:-1] = knotvals[n:]
        asymmetry_parameter[-1] = 0.5 * knotvals[-1]
        self.estimate = EmissivityProfile(
            knots, symmetric_emissivity, asymmetry_parameter, self.flux_coords, 45.0
        )
        print(self.estimate(self.flux_coords, None, None))

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
                residuals, guess, args=(rhos, Rs, t, R_0s, dls), verbose=2
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
            results.append(self.estimate(self.flux_coords, None, None))
            guess = fit.x
        result = concat(results, dim=self.t)
        result.attrs["transform"] = self.flux_coords
        result.attrs["datatype"] = ("emissivity", "sxr")
        result.attrs["provenance"] = self.create_provenance()
        result.name = "sxr_emissivity"
        return result
