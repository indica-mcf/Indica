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
from xarray import Dataset
from xarray import where

from .abstractoperator import Operator
from ..converters import bin_to_time_labels
from ..converters import CoordinateTransform
from ..converters import FluxMajorRadCoordinates
from ..converters import FluxSurfaceCoordinates
from ..converters import ImpactParameterCoordinates
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
                (1, np.zeros_like(symmetric_emissivity.isel({self._dim: 0}))),
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
    n_intervals : int
        The number of intervals over which to integrate th eemissivity. Should
        be 2 ** m + 1, where m is an integer.
    sess : Session
        An object representing the session being run. Contains information
        such as provenance data.

    """

    RETURN_TYPES: List[DataType] = [("emissivity", "sxr")]

    def __init__(
        self,
        num_cameras: int = 1,
        n_knots: int = 6,
        n_intervals: int = 65,
        sess: Session = global_session,
    ):
        self.n_knots = n_knots
        self.n_intervals = n_intervals
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
    ) -> Dataset:
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
        binned_cameras = [bin_to_time_labels(times.data, c) for c in cameras]

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
            unfolded_cameras: List[Dataset],
        ) -> np.ndarray:
            symmetric_emissivity, asymmetry_parameter = knotvals_to_xarray(knotvals)
            estimate = EmissivityProfile(
                symmetric_emissivity, asymmetry_parameter, flux_coords
            )
            start = 0
            resid = np.empty(sum(c.attrs["nlos"] for c in unfolded_cameras))
            self.integral = []
            for c in unfolded_cameras:
                end = start + c.attrs["nlos"]
                rho, R = c.indica.convert_coords(rho_maj_rad)
                rho_min, x2 = c.indica.convert_coords(c.attrs["impact_parameters"])
                emissivity_vals = estimate.evaluate(rho, R, R_0=c.coords["R_0"])
                axis = emissivity_vals.dims.index(c.attrs["transform"].x2_name)
                integral = romb(emissivity_vals, c.attrs["dl"], axis)
                resid[start:end] = ((c.camera - integral) / c.weights).data
                start = end
                x1_name = c.attrs["transform"].x1_name
                self.integral.append(
                    DataArray(integral, coords=[(x1_name, c.coords[x1_name])])
                )
            assert np.all(np.isfinite(resid))
            return resid

        x2 = np.linspace(0.0, 1.0, self.n_intervals)
        unfolded_cameras = [
            Dataset(
                {"camera": bin_to_time_labels(times.data, c)},
                {c.attrs["transform"].x2_name: x2},
                {"transform": c.attrs["transform"]},
            )
            for c in binned_cameras
        ]

        rho_maj_rad = FluxMajorRadCoordinates(flux_coords)
        rho_max = 0.0
        print("Calculating coordinate conversions")
        for c in unfolded_cameras:
            trans = c.attrs["transform"]
            dl = trans.distance(
                trans.x2_name, DataArray(0), c.coords[trans.x2_name][0:2], 0.0
            )[1]
            c.attrs["dl"] = dl
            c.attrs["nlos"] = len(c.coords[c.attrs["transform"].x1_name])
            rho, _ = c.indica.convert_coords(rho_maj_rad)
            ip_coords = ImpactParameterCoordinates(
                c.attrs["transform"], flux_coords, times=times
            )
            c.attrs["impact_parameters"] = ip_coords
            rho_max = max(rho_max, ip_coords.rhomax())
            impact_param, _ = c.indica.convert_coords(ip_coords)
            c["weights"] = c.camera * (0.02 + 0.18 * np.abs(impact_param))
            c["weights"].attrs["transform"] = c.camera.attrs["transform"]
            c.coords["R_0"] = c.attrs["transform"].equilibrium.R_hfs(
                rho, c.coords["t"]
            )[0]

        knots = np.empty(n)
        knots[0 : n - 1] = np.linspace(0, 1.0, n - 1) ** 1.2 * float(rho_max)
        knots[-1] = 1.0
        dim_name = "rho_" + flux_coords.flux_kind

        symmetric_emissivities: List[DataArray] = []
        asymmetry_parameters: List[DataArray] = []
        integrals: List[List[DataArray]] = []
        abel_inversion = np.linspace(3e3, 0.0, n - 1)
        guess = np.concatenate((abel_inversion, np.zeros(n - 2)))
        bounds = (
            np.concatenate((np.zeros(n - 1), np.where(knots[1:-1] > 0.5, 0.0, -0.5))),
            np.concatenate((1e6 * np.ones(n - 1), np.ones(n - 2))),
        )

        for t in np.asarray(times):
            print(f"\nSolving for t={t}")
            print("------------------\n")
            fit = least_squares(
                residuals,
                guess,
                bounds=bounds,
                args=([c.sel(t=t) for c in unfolded_cameras],),
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
        symmetric_emissivity.attrs["transform"] = flux_coords
        asymmetry_parameter = concat(asymmetry_parameters, dim=times)
        asymmetry_parameter.attrs["transform"] = flux_coords
        integral: List[DataArray] = []
        for data in zip(*integrals):
            integral.append(concat(data, dim=times))
            del integral[-1].coords[None]  # type: ignore
        # For some reason concat adds a `None` coordinate
        del symmetric_emissivity.coords[None]
        del asymmetry_parameter.coords[None]
        estimate = EmissivityProfile(
            symmetric_emissivity, asymmetry_parameter, flux_coords
        )
        trivial = TrivialTransform()
        emissivity = estimate(trivial, R, z, times)
        emissivity.attrs["transform"] = trivial
        emissivity.attrs["datatype"] = ("emissivity", "sxr")
        emissivity.attrs["provenance"] = self.create_provenance()
        emissivity.name = "sxr_emissivity"
        results = {
            "emissivity": emissivity,
            "symmetric_emissivity": symmetric_emissivity,
            "asymmetry_parameter": asymmetry_parameter,
        }
        for inte, c, c_orig in zip(integral, unfolded_cameras, cameras):
            name = c_orig.name
            results[name + "_binned"] = c.camera
            inte.attrs["transform"] = c.camera.attrs["transform"]
            results[name + "_back_integral"] = inte
            results[name + "_weights"] = c.weights
        return Dataset(results, c.coords, {"emissivity_model": estimate})
