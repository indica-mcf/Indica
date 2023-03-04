"""Inverts soft X-ray data to estimate the emissivity of the plasma."""

from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

import numpy as np
from scipy.integrate import romb
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from xarray import concat
from xarray import DataArray
from xarray import Dataset
from xarray import where
import time as tt
from scipy import interpolate

from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..converters import bin_to_time_labels
from ..converters import CoordinateTransform
from ..converters import FluxMajorRadCoordinates
from ..converters import FluxSurfaceCoordinates
from ..converters import ImpactParameterCoordinates
from ..converters import TrivialTransform
from ..datatypes import DataType
from ..datatypes import SpecificDataType
from ..utilities import broadcast_spline

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
        transpose_order = (self._dim,) + self.asym_dims
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
        """Get the emissivity values at the locations given by the coordinates.

        Parameters
        ----------
        coord_system
            The transform describing the system used by the provided coordinates
        x1
            The first spatial coordinate
        x2
            The second spatial coordinate
        t
            The time coordinate

        """
        rho, R = cast(
            DataArrayCoords, coord_system.convert_to(self.transform, x1, x2, t)
        )
        return self.evaluate(rho, R, t).assign_attrs(transform=coord_system)

    def evaluate(
        self,
        rho: DataArray,
        R: DataArray,
        t: Optional[DataArray] = None,
        R_0: Optional[DataArray] = None,
    ) -> DataArray:
        """Evaluate the function at a location defined using (R, z) coordinates"""
        if t is None:
            if "t" in rho.coords:
                t = rho.coords["t"]
            elif self.time is not None:
                t = self.time
        elif "t" not in rho.coords and (
            "t" in self.sym_coords or "t" in self.asym_coords
        ):
            rho = rho.expand_dims(t=t)
        symmetric = broadcast_spline(
            self.symmetric_emissivity, self.sym_dims, self.sym_coords, rho
        )
        asymmetric = broadcast_spline(
            self.asymmetry_parameter, self.asym_dims, self.asym_coords, rho
        )
        if R_0 is None:
            R_0 = cast(DataArray, self.transform.equilibrium.R_hfs(rho, t)[0])
        result = symmetric * np.exp(asymmetric * (R ** 2 - R_0 ** 2))
        # Ensure round-off error doesn't result in any values below 0
        return where(result < 0.0, 0.0, result).fillna(0.0)


class InvertRadiation(Operator):
    """Estimates the emissivity distribution of the plasma using radiation
    data.

    Parameters
    ----------
    flux_coordinates : FluxSurfaceCoordinates
        The flux surface coordinate system on which to calculate the fit.
    datatype : SpecificDataType
        The type of radiation data to be inverted.
    num_cameras : int
        The number of cameras to which the data will be fit.
    n_knots : int
        The number of  spline knots to use when fitting the emissivity data.
    n_intervals : int
        The number of intervals over which to integrate th eemissivity. Should
        be :math:`2^m + 1`, where m is an integer.
    sess : session.Session
        An object representing the session being run. Contains information
        such as provenance data.

    """

    def __init__(
        self,
        num_cameras: int = 1,
        datatype: SpecificDataType = "sxr",
        n_knots: int = 6,
        run_parallel: bool = False,
        fit_asymmetry: bool = False,
        debug: bool = False,
        sess: session.Session = session.global_session,
    ):
        self.n_knots = n_knots
        self.datatype = datatype
        self.num_cameras = num_cameras
        self.integral: List[DataArray]
        self.last_knot_zero = datatype == "sxr"
        # TODO: Update RETURN_TYPES
        # TODO: Revise to include R, z, t
        self.ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = [
            ("major_rad", None),
            ("z", None),
            ("time", None),
            ("luminous_flux", datatype),
            ...,
        ]
        super().__init__(
            sess, num_cameras=num_cameras, datatype=datatype, n_knots=n_knots,
        )
        self.debug = debug
        self.parallel_run = run_parallel
        self.fit_asymmetry = fit_asymmetry

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        """Indicates the datatypes of the results when calling the operator
        with arguments of the given types. It is assumed that the
        argument types are valid.

        Parameters
        ----------
        args
            The datatypes of the parameters which the operator is to be called with.

        Returns
        -------
        :
            The datatype of each result that will be returned if the operator is
            called with these arguments.

        """
        radiation = cast(DataType, self.ARGUMENT_TYPES[-1])[1]
        result = cast(
            Tuple[DataType, ...],
            (
                ("emissivity", radiation),
                (
                    radiation,
                    {
                        "symmetric_emissivity": "emissivity",
                        "asymmetry_parameter": "asymmetry",
                    },
                ),
            ),
        ) + cast(
            Tuple[DataType, ...],
            (
                (
                    radiation,
                    {
                        "camera": "luminous_flux",
                        "weights": "weighting",
                        "back_integral": "luminous_flux",
                    },
                ),
            ),
        ) * (
            len(args) - 3
        )
        return result

    @staticmethod
    def knot_positions(n: int, rho_max: float):
        """Calculates location of knots in magnetic flux coordinates.

        Parameters
        ----------
        n
            The number of knots needed.
        rho_max
            The normalised magnetic flux of the final knot location.
        """
        knots = np.empty(n)
        if float(rho_max) > 1.0:
            knots[:] = np.linspace(0, 1.0, n) ** 1.2 * float(rho_max)
        else:
            knots[0 : n - 1] = np.linspace(0, 1.0, n - 1) ** 1.2 * float(rho_max)
            knots[-1] = 1.0
        return knots

    def __call__(  # type: ignore[override]
        self, R: DataArray, z: DataArray, times: DataArray, *cameras: DataArray,
    ) -> Tuple[Union[DataArray, Dataset], ...]:
        """Calculate the emissivity profile for the plasma.

        Parameters
        ----------
        R
            Major radii on which to return emissivity result.
        z
            Theta coordinates on which to return emissivity result.
        times
            Time coordinatse on which to return emissivity result.
        cameras
            The binned luminosity data being fit to,.

        Returns
        -------
        : DataArray
            The fit emissivity, on the R-z grid. Will also contain an
            attribute "emissivity_model", which is an
            :py:class:`indica.operators.invert_radiation.EmissivityProfile`
            object that can interpolate the fit emissivity onto
            arbitrary coordinates.
        : Dataset
            A dataset containing

            - **symmetric_emissivity**: The symmetric emissivity
               values which were found during the fit, given along
               :math:`\\rho`.
            - **asymmetry_parameter**: The asymmetry of the emissivity
              which was found during the fit, given along :math:`\\rho`.

        : Dataset
            For each camera passed as an argument, a dataset containing

            - **camera**: The radiation data for that camera, binned in time.
            - **back_integral**: The integral of the fit emissivity along
              the lines of sight of the camera.
            - **weights**: The weights assigned to each line of sight of the
              camera when fitting emissivity.

        """
        if self.debug:
            debug_data = {"invert_class": {}}
            start_time = tt.time()
            st = start_time

        # # self.validate_arguments(R, z, times, *cameras)
        n = self.n_knots
        knots = self.knot_positions(n, self.rho_max)
        flux_coords = cameras[1]

        symmetric_emissivities: List[DataArray] = []
        asymmetry_parameters: List[DataArray] = []
        integrals: List[List[DataArray]] = []
        m = n - 1 if self.last_knot_zero else n
        abel_inversion = np.linspace(3e3, 0.0, m)
        if self.fit_asymmetry:
            guess = np.concatenate((abel_inversion, np.zeros(n - 2)))
            bounds = (
                np.concatenate((np.zeros(m), np.zeros(n - 2))),
                np.concatenate((1e12 * np.ones(m), 1.0e-20 * np.ones(n - 2))),
            )
        else:
            guess = abel_inversion
            bounds = (
                np.concatenate((np.zeros(m), np.zeros(0))),
                np.concatenate((1e12 * np.ones(m), 1.0e-20 * np.ones(0))),
            )
        # DEBUG TIME
        if self.debug:
            step = "Initializing guesses"
            step_time = np.round(tt.time() - st, 2)
            debug_data["invert_class"][step] = step_time
            print(step + ". It took " + str(step_time) + " seconds")
            st = tt.time()

        # INPUT DATA
        input_data = dict(guess=guess, bounds=bounds, unfolded_cameras=cameras[0])

        # FUNCTION TO CONVERT KNOT POINTS TO SYMMETRIC EMISSIVITY AND ASYMMETRY PARAMETER
        def knotvals_to_xarray(knotvals):
            symmetric_emissivity = DataArray(
                np.empty(n), coords=[(self.dim_name, knots)]
            )
            symmetric_emissivity[0:m] = knotvals[0:m]
            if self.last_knot_zero:
                symmetric_emissivity[-1] = 0.0
            asymmetry_parameter = DataArray(
                np.zeros(n), coords=[(self.dim_name, knots)]
            )
            if self.fit_asymmetry:
                asymmetry_parameter[0] = 0.5 * knotvals[m]
                asymmetry_parameter[1:-1] = knotvals[m:]
                asymmetry_parameter[-1] = 0.5 * knotvals[-1]
            return symmetric_emissivity, asymmetry_parameter

        # RESIDUAL
        def residuals(
            knotvals: np.ndarray,
            unfolded_cameras: List[Dataset],
            compute_integral: bool = False,
        ) -> np.ndarray:
            symmetric_emissivity, asymmetry_parameter = knotvals_to_xarray(knotvals)
            estimate = EmissivityProfile(
                symmetric_emissivity, asymmetry_parameter, flux_coords
            )
            start = 0
            resid = np.empty(sum(c.attrs["nlos"] for c in unfolded_cameras))
            integrals = []
            for c in unfolded_cameras:
                end = start + c.attrs["nlos"]
                #     rho, R = c.indica.convert_coords(rho_maj_rad)
                #     rho_min, x2 = c.indica.convert_coords(c.attrs["impact_parameters"])
                emissivity_vals = estimate.evaluate(
                    c.coords["rho"], c.coords["R"], R_0=c.coords["R_0"]
                )
                axis = emissivity_vals.dims.index(c.attrs["transform"].x2_name)
                integral = romb(emissivity_vals, c.attrs["dl"], axis)
                resid[start:end] = ((c.camera - integral) / c.weights)[
                    c["has_data"]
                ].data
                start = end
                x1_name = c.attrs["transform"].x1_name
                if compute_integral:
                    integrals.append(
                        DataArray(integral, coords=[(x1_name, c.coords[x1_name])])
                    )
            if compute_integral:
                return integrals
            else:
                assert np.all(np.isfinite(resid))
                return resid

        # FIT FUNCTION
        def fit_function(t, input_data):
            fit = least_squares(
                residuals,
                input_data["guess"],
                bounds=input_data["bounds"],
                args=([c.sel(t=t) for c in input_data["unfolded_cameras"].values()],),
                verbose=0,
            )
            return fit

        # SWEEP OF TIMES
        for it, t in enumerate(np.asarray(times)):
            # LEAST SQUARED FIT
            if not self.parallel_run:
                fit = fit_function(t, input_data)
            # FIT STATUS
            if fit.status == -1:
                raise RuntimeError(
                    "Improper input to `least_squares` function when trying to "
                    "fit emissivity to radiation data."
                )
            elif fit.status == 0:
                warnings.warn(
                    f"Attempt to fit emissivity to radiation data at time t={t} "
                    "reached maximum number of function evaluations.",
                    RuntimeWarning,
                )
            sym, asym = knotvals_to_xarray(fit.x)
            if not self.fit_asymmetry:
                try:
                    asym.data = interpolate.interp1d(
                        self.asymmetry_parameter.p.data,
                        self.asymmetry_parameter.sel(t=t).data,
                    )(knots)
                except:
                    pass

            symmetric_emissivities.append(sym)
            asymmetry_parameters.append(asym)
            integrals.append(
                residuals(
                    fit.x,
                    [c.sel(t=t) for c in input_data["unfolded_cameras"].values()],
                    True,
                )
            )
            if not self.parallel_run:
                input_data["guess"] = fit.x.copy()

        # DEBUG TIME
        if self.debug:
            step = "Fitting SXR data"
            step_time = np.round(tt.time() - st, 2)
            debug_data["invert_class"][step] = step_time
            print(step + ". It took " + str(step_time) + " seconds")
            st = tt.time()

        # FINAL TRANSFORMATION
        symmetric_emissivity = concat(symmetric_emissivities, dim=times)
        symmetric_emissivity.attrs["transform"] = flux_coords
        symmetric_emissivity.attrs["datatype"] = ("emissivity", self.datatype)
        asymmetry_parameter = concat(asymmetry_parameters, dim=times)
        asymmetry_parameter.attrs["transform"] = flux_coords
        asymmetry_parameter.attrs["datatype"] = ("asymmetry", self.datatype)
        # For some reason concat adds a `None` coordinate
        del symmetric_emissivity.coords[None]
        del asymmetry_parameter.coords[None]
        # BACK INTEGRAL VALUES
        back_integrals = {}
        for key in cameras[0].keys():
            back_integrals[key] = np.nan * np.ones(
                (cameras[0][key]["camera"].data.shape)
            )
            integrals = np.array(integrals)
            # SWEEP OF TIMES
            for iback in range(0, np.size(back_integrals[key], 0)):
                temp_data = integrals[iback][0].data
                back_integrals[key][iback, :] = temp_data

        # 2D EMISSIVITY PROFILE
        estimate = EmissivityProfile(
            symmetric_emissivity, asymmetry_parameter, flux_coords
        )
        trivial = TrivialTransform()
        emissivity = estimate(trivial, R, z, times)
        emissivity.attrs["datatype"] = ("emissivity", self.datatype)
        emissivity.attrs["emissivity_model"] = estimate
        emissivity.name = self.datatype + "_emissivity"

        # DEBUG TIME
        if self.debug:
            step = "Final transformation"
            step_time = np.round(tt.time() - st, 2)
            debug_data["invert_class"][step] = step_time
            print(step + ". It took " + str(step_time) + " seconds")
            st = tt.time()

        # GATHERING THE DATA
        return_data = {}

        # SWEEP OF CAMERAS
        for key in cameras[0].keys():
            return_data[key] = dict(
                t=times.data,
                channels_considered=cameras[0][key].has_data.data,
                # BACK INTEGRAL DATA
                back_integral=dict(
                    p_impact=np.round(
                        cameras[0][key].attrs["impact_parameters"].rho_min.data, 2
                    ).T,
                    data_experiment=cameras[0][key]["camera"].data,
                    data_theory=back_integrals[key],
                    channel_no=np.arange(
                        1, np.size(cameras[0][key]["camera"].data, 1) + 1
                    ),
                ),
                # PROJECTION DATA
                projection=dict(R=cameras[0][key].R.data, z=cameras[0][key].z.data,),
                # PROFILES
                profile=dict(
                    sym_emissivity=symmetric_emissivity.data,
                    asym_parameter=asymmetry_parameter.data,
                    rho_poloidal=np.repeat(
                        np.array([asymmetry_parameter.rho_poloidal.data]),
                        len(times.data),
                        axis=0,
                    ),
                ),
                # EMISSIVITY 2D
                emissivity_2D=dict(
                    R=emissivity.R.data, z=emissivity.z.data, data=emissivity.data.T,
                ),
            )

            # ESTIMATING THE CHI2
            data_exp = return_data[key]["back_integral"]["data_experiment"][
                :, return_data[key]["channels_considered"]
            ]
            data_the = return_data[key]["back_integral"]["data_theory"][
                :, return_data[key]["channels_considered"]
            ]
            return_data[key]["back_integral"]["chi2"] = np.sqrt(
                np.nansum(((data_exp - data_the) ** 2) / (data_exp ** 2), axis=1)
            )

            print("hahahaha", data_exp[0, :], data_the[0, :], data_exp.shape)

        # APPENDING DEBUG DATA
        if self.debug:
            return_data["debug_data"] = debug_data

        # RETURNING THE DATA
        return return_data
