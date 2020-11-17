"""Inverts soft X-ray data to estimate the emissivity of the plasma."""

from typing import cast
from typing import List
from typing import Tuple
from typing import Union
import warnings

import numpy as np
from scipy.integrate import romb
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from xarray import concat
from xarray import DataArray

from .abstractoperator import Operator
from ..converters import FluxMajorRadCoordinates
from ..converters import FluxSurfaceCoordinates
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
        self.beta = 1.0
        self.coefficient = self.emissivity_rho_95 / (np.exp(-self.beta * 0.05) - 1.0)
        self.time = time

    def __call__(self, R: DataArray, z: DataArray) -> DataArray:
        rho, R, _ = cast(
            DataArrayCoords, self.transform.convert_from_Rz(R, z, self.time)
        )
        return self.evaluate(rho, R)

    def evaluate(self, rho: DataArray, R: DataArray) -> DataArray:
        """Evaluate the function at a location defined using (R, z) coordinates
        """
        R_0, _ = self.transform.equilibrium.R_hfs(rho, self.time)
        symmetric = self.symmetric_emissivity(rho)
        mask = np.logical_and(rho > 0.95, rho < 1.0)
        if np.any(mask):
            symmetric[mask] = self.emissivity_rho_95 + self.coefficient * (
                1 - np.exp(-self.beta * (rho[mask] - 0.95))
            )
        mask = rho >= 1.0
        if np.any(mask):
            symmetric[mask] = 0.0
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
    n = 2 ** np.ceil(np.log(2 - 1) / np.log(2)) + 1
    x2 = DataArray(np.linspace(0.0, 1.0, n), dims="x2")
    distances, _ = los.distance(2, DataArray(0), x2[0:2], emissivity.time)
    dl = cast(DataArray, distances)[1]
    R, z, _ = cast(DataArrayCoords, los.convert_to_Rz(x2=x2, t=emissivity.time))
    emissivity_vals = emissivity(R, z)
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
        self.R, self.z, self.t = cast(DataArrayCoords, flux_coordinates.convert_to_Rz())
        self.num_cameras = num_cameras
        self.ARGUMENT_TYPES: List[DataType] = [("luminous_flux", "sxr")] * num_cameras
        self.estimate: EmissivityProfile
        super().__init__(
            sess,
            flux_coordinates=flux_coordinates,
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
        weight_a_param = 0.18 / np.expm1(5)
        weight_b_param = 0.2 - weight_a_param
        num_los = [len(c.attrs["transform"].default_x1) for c in cameras]
        weights = [
            weight_a_param * np.exp(np.linspace(0, 5, n_los)) + weight_b_param
            for n_los in num_los
        ]
        n = 8
        knots = np.linspace(0.0, 1.0, n)

        def residuals(knotvals: np.ndarray, time: DataArray) -> np.ndarray:
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
            for camera, n_los, weight in zip(cameras, num_los, weights):
                end = start + n_los
                resid[start:end] = (
                    integrate_los(camera.attrs["transform"], self.estimate).data
                    / weights
                )
                start = end
            return resid

        results: List[DataArray] = []
        abel_inversion = np.zeros(n)
        guess = np.concatenate((abel_inversion, np.zeros(n - 2)))
        for t in np.asarray(self.t):
            fit = least_squares(residuals, guess, args=(t,))
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
            results.append(self.estimate(self.R.sel(t=t), self.z.sel(t=t)))
            guess = fit.x
        result = concat(results, dim=self.t)
        result.attrs["transform"] = self.flux_coords
        result.attrs["datatype"] = ("emissivity", "sxr")
        result.attrs["provenance"] = self.create_provenance()
        result.name = "sxr_emissivity"
        return result
