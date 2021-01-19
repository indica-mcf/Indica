"""Inverts soft X-ray data to estimate the emissivity of the plasma."""

from typing import Any
from typing import cast
from typing import Dict
from typing import Hashable
from typing import List
from typing import Optional
from typing import Tuple
import warnings

import numpy as np
from scipy.integrate import romb
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from xarray import apply_ufunc
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
from ..converters import TrivialTransform
from ..datatypes import DataType
from ..session import global_session
from ..session import Session

DataArrayCoords = Tuple[DataArray, DataArray]


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
        symmetric_emissivity: DataArray,
        asymmetry_parameter: DataArray,
        coord_transform: FluxSurfaceCoordinates,
    ):
        self._dim = coord_transform.x1_name
        if self._dim not in symmetric_emissivity.dims:
            raise ValueError(
                "`symmetric_emissivity` must have the same x1 coordinate as "
                "`coord_transform`."
            )
        if self._dim not in asymmetry_parameter.dims:
            raise ValueError(
                "`symmetric_emissivity` must have the same x1 coordinate as "
                "`coord_transform`."
            )
        self.sym_dims = tuple(d for d in symmetric_emissivity.dims if d != self._dim)
        self.sym_coords = {
            k: np.asarray(v)
            for k, v in symmetric_emissivity.coords.items()
            if k != self._dim
        }
        transpose_order = (self._dim,) + self.sym_dims
        self.symmetric_emissivity = CubicSpline(
            symmetric_emissivity.coords[self._dim],
            symmetric_emissivity.transpose(*transpose_order),
            0,
            (
                (2, np.zeros_like(symmetric_emissivity.isel({self._dim: 0}))),
                (2, np.zeros_like(symmetric_emissivity.isel({self._dim: -1}))),
            ),
            False,
        )
        self.asym_dims = tuple(d for d in asymmetry_parameter.dims if d != self._dim)
        self.asym_coords = {
            k: np.asarray(v)
            for k, v in asymmetry_parameter.coords.items()
            if k != self._dim
        }
        transpose_order = (self._dim,) + self.sym_dims
        self.asymmetry_parameter = CubicSpline(
            asymmetry_parameter.coords[self._dim],
            asymmetry_parameter.transpose(*transpose_order),
            0,
            (
                (2, np.zeros_like(asymmetry_parameter.isel({self._dim: 0}))),
                (2, np.zeros_like(asymmetry_parameter.isel({self._dim: -1}))),
            ),
        )
        self.transform = FluxMajorRadCoordinates(coord_transform)
        time_sym = symmetric_emissivity.coords.get("t", None)
        time_asym = asymmetry_parameter.coords.get("t", None)
        if (
            time_sym is not None
            and time_asym is not None
            and not time_sym.equals(time_asym)
        ):
            raise ValueError(
                "Dimension 't' must match for `symmetric_emissivity` and "
                "`asymmetry_parameter`"
            )
        self.time = time_sym if time_sym is not None else time_asym

    def __call__(
        self,
        coord_system: CoordinateTransform,
        x1: DataArray,
        x2: DataArray,
        t: DataArray,
    ) -> DataArray:
        rho, R = cast(
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

        def broadcast_spline(
            spline: CubicSpline,
            spline_dims: Tuple,
            spline_coords: Dict[Hashable, Any],
        ):
            if "t" in rho.coords and "t" in spline_dims:
                time_outer_product = apply_ufunc(
                    spline,
                    rho,
                    input_core_dims=[[]],
                    output_core_dims=[
                        tuple(d if d != "t" else "__new_t" for d in spline_dims)
                    ],
                ).assign_coords(__new_t=spline_coords["t"])
                result = time_outer_product.indica.interp2d(
                    __new_t=rho.coords["t"], method="cubic"
                )
                del result.coords["__new_t"]
            else:
                result = apply_ufunc(
                    spline,
                    rho,
                    input_core_dims=[[]],
                    output_core_dims=[spline_dims],
                )
            return result.assign_coords(
                {k: v for k, v in spline_coords.items() if k != "t"}
            )

        symmetric = broadcast_spline(
            self.symmetric_emissivity, self.sym_dims, self.sym_coords
        )
        asymmetric = broadcast_spline(
            self.asymmetry_parameter, self.asym_dims, self.asym_coords
        )
        if t is None:
            if "t" in rho.coords:
                t = rho.coords["t"]
            elif self.time is not None:
                t = self.time
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
    num_cameras : int
        The number of cameras to which the data will be fit.
    n_knots : int
        The number of  spline knots to use when fitting the emissivity data.
    sess : Session
        An object representing the session being run. Contains information
        such as provenance data.

    """

    RETURN_TYPES: List[DataType] = [("emissivity", "sxr")]

    def __init__(
        self,
        num_cameras: int = 1,
        n_knots: int = 6,
        sess: Session = global_session,
    ):
        self.n_knots = n_knots
        self.num_cameras = num_cameras
        self.integral: List[DataArray]
        self.ARGUMENT_TYPES: List[DataType] = [("luminous_flux", "sxr")] * num_cameras
        super().__init__(
            sess,
            num_cameras=num_cameras,
            n_knots=n_knots,
        )

    def __call__(  # type: ignore[override]
        self,
        R: DataArray,
        z: DataArray,
        times: DataArray,
        *cameras: DataArray,
    ) -> DataArray:
        """Calculate the emissivity profile for the plasma.

        Parameters
        ----------
        R : DataArray
            Major radii on which to return emissivity result.
        z : DataArray
            Theta coordinates on which to return emissivity result.
        t : DataArray
            Time coordinatse on which to return emissivity result.
        cameras
            The luminosity data being fit to, with each camera passed
            as a separate argument.

        Returns
        -------
        :
            Emissivity values on the (\rho, \theta) grid specified when
            this operator was constructed.
        """
        flux_coords = FluxSurfaceCoordinates("poloidal")
        flux_coords.set_equilibrium(cameras[0].attrs["transform"].equilibrium)
        self.validate_arguments(*cameras)
        n = self.n_knots
        num_los = [len(c.coords[c.attrs["transform"].x1_name]) for c in cameras]
        print("Calculating impact parameters")
        impact_params = [
            ImpactParameterCoordinates(c.attrs["transform"], flux_coords, times=times)
            for c in cameras
        ]
        binned_cameras = [bin_to_time_labels(times.data, c) for c in cameras]
        print("...done")
        rho_max = max(ip.rhomax() for ip in impact_params)
        knots = np.empty(n)
        knots[0 : n - 1] = np.linspace(0, 1.0, n - 1) ** 1.2 * float(rho_max)
        knots[-1] = 1.0
        dim_name = "rho_" + flux_coords.flux_kind

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
                symmetric_emissivity, asymmetry_parameter, flux_coords
            )
            start = 0
            resid = np.empty(sum(num_los))
            self.integral = []
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
                        * (0.02 + 0.18 * np.abs(ip_coords.rho_min.sel(t=time)))
                    )
                ).data
                start = end
                x1_name = camera.attrs["transform"].x1_name
                self.integral.append(
                    DataArray(integral, coords=[(x1_name, camera.coords[x1_name])])
                )
            assert np.all(np.isfinite(resid))
            return resid

        x2 = DataArray(np.linspace(0.0, 1.0, 65), dims="x2")
        dls = [
            cast(
                DataArray,
                c.attrs["transform"].distance("x2", DataArray(0), x2[0:2], 0.0),
            )[1]
            for c in cameras
        ]

        symmetric_emissivities: List[DataArray] = []
        asymmetry_parameters: List[DataArray] = []
        integrals: List[List[DataArray]] = []
        abel_inversion = np.linspace(3e3, 0.0, n - 1)
        guess = np.concatenate((abel_inversion, np.zeros(n - 2)))
        bounds = (
            np.concatenate((np.zeros(n - 1), -0.5 * np.ones(n - 2))),
            np.concatenate((1e6 * np.ones(n - 1), np.ones(n - 2))),
        )
        rho_maj_rad = FluxMajorRadCoordinates(flux_coords)

        for t in np.asarray(times):
            print(f"Solving for t={t}")
            print("-----------------\n")
            rhos, Rs = zip(
                *[
                    c.attrs["transform"].convert_to(
                        rho_maj_rad, c.coords[c.attrs["transform"].x1_name], x2, t
                    )
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
            integrals.append(self.integral)
            guess = fit.x
        symmetric_emissivity = concat(symmetric_emissivities, dim=times)
        asymmetry_parameter = concat(asymmetry_parameters, dim=times)
        integral: List[DataArray] = []
        for data in zip(*integrals):
            integral.append(concat(data, dim=times))
            del integral[-1].coords[None]
        # For some reason concat adds a `None` coordinate
        del symmetric_emissivity.coords[None]
        del asymmetry_parameter.coords[None]
        estimate = EmissivityProfile(
            symmetric_emissivity, asymmetry_parameter, flux_coords
        )
        trivial = TrivialTransform()
        result = estimate(trivial, R, z, times)
        result.attrs["transform"] = trivial
        result.attrs["datatype"] = ("emissivity", "sxr")
        result.attrs["provenance"] = self.create_provenance()
        result.attrs["emissivity"] = estimate
        result.name = "sxr_emissivity"
        return result, *integral
