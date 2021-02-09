"""Fit 1-D splines to data."""

from itertools import accumulate
from itertools import chain
from itertools import repeat
from itertools import tee
from typing import cast
from typing import Hashable
from typing import List
from typing import Tuple
from typing import Union
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from xarray import DataArray

from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from ..converters import bin_to_time_labels
from ..converters import CoordinateTransform
from ..converters import FluxSurfaceCoordinates
from ..datatypes import DataType
from ..numpy_typing import ArrayLike
from ..session import global_session
from ..session import Session
from ..utilities import broadcast_spline
from ..utilities import coord_array


SingleBoundaryType = Union[str, Tuple[int, ArrayLike]]
BoundaryType = Union[str, Tuple[SingleBoundaryType, SingleBoundaryType]]


class Spline:
    """Callable class wrapping a `:class:scipy.interpolate.CubicSpline`
    object so it will work with DataArrays.

    Parameters
    ----------
    values : DataArray
        The values to interpolate.
    dim : Hashable
        The axis along which to interpolate.
    coord_transform : CoordinateTransform
        The transform describing the coordinate system used by `values`.
    bounds : BoundaryType
        The boundary condition to pass to `:class:scipy.interpolate.CubicSpline`.

    """

    def __init__(
        self,
        values: DataArray,
        dim: Hashable,
        coord_transform: CoordinateTransform,
        bounds: BoundaryType = "clamped",
    ):
        self.dim = dim
        self.spline_dims = tuple(d for d in values.dims if d != dim)
        self.spline_coords = {
            k: np.asarray(v) for k, v in values.coords.items() if k != self.dim
        }
        transpose_order = (self.dim,) + self.spline_dims
        self.spline = CubicSpline(
            values.coords[dim], values.transpose(*transpose_order), 0, bounds, False
        )
        self.transform = coord_transform

    def __call__(
        self,
        coord_system: CoordinateTransform,
        x1: DataArray,
        x2: DataArray,
        t: DataArray,
    ) -> DataArray:
        """Get the spline values at the locations given by the coordinates.

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
        self_x1, self_x2 = cast(
            Tuple[DataArray, DataArray],
            coord_system.convert_to(self.transform, x1, x2, t),
        )
        coord = self_x1 if self.dim == self.transform.x1_name else self_x2
        result = broadcast_spline(
            self.spline, self.spline_dims, self.spline_coords, coord
        )
        result.attrs["transform"] = coord_system
        return result


class SplineFit(Operator):
    """Fit a 1-D spline to data. The spline will be given on poloidal flux
    surface coordinates, as specified by the user. It can derive a
    single spline fit for multiple DataArray arguments simultaneously.

    Parameters
    ----------
    knots : ArrayLike
        A 1-D array containing the location of spline knots to use when
        fitting the data.
    lower_bound : ArrayLike
        The lower bounds to use for values at each not. May be either a
        scalar or an array of the same shape as ``knots``.
    upper_bound : ArrayLike
        The upper bounds to use for values at each not. May be either a
        scalar or an array of the same shape as ``knots``.
    sess : Session
        An object representing the session being run. Contains information
        such as provenance data.

    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("rho_poloidal", "plasma"),
        ("time", "plasma"),
        (None, None),
        ...,
    ]

    def __init__(
        self,
        knots: ArrayLike = [0.0, 0.3, 0.6, 0.85, 0.95, 1.05],
        lower_bound: ArrayLike = -np.inf,
        upper_bound: ArrayLike = np.inf,
        sess: Session = global_session,
    ):
        self.knots = coord_array(knots, "rho_poloidal")
        self.lower_bound = lower_bound
        if isinstance(lower_bound, np.ndarray) and lower_bound.size != self.knots.size:
            raise ValueError(
                "lower_bound must be either a scalar or array of same size as knots"
            )
        self.upper_bound = upper_bound
        if isinstance(upper_bound, np.ndarray) and upper_bound.size != self.knots.size:
            raise ValueError(
                "lower_bound must be either a scalar or array of same size as knots"
            )
        self.spline: Spline
        self.spline_vals: DataArray
        super().__init__(
            sess,
            knots=str(knots),
            lower_bound=str(lower_bound),
            upper_bound=str(upper_bound),
        )

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
        input_type = args[-1]
        return (input_type,) * len(args)

    def __call__(  # type: ignore[override]
        self,
        rho: DataArray,
        times: DataArray,
        *data: DataArray,
    ) -> Tuple[DataArray, ...]:
        """Fit a spline to the provided data.

        Parameters
        ----------
        rho
            The poloidal flux values on which to return the result.
        times
            The times at which to bin the data and return the result.
        data
            The data to fit the spline to.

        Returns
        -------
        :
            The results of the fit on the give \\rho and time values.
            It contains the attribute `splines` which can be used to
            interpolate results onto arbitrary coordinates.

        """
        n_knots = len(self.knots)
        flux_surfaces = FluxSurfaceCoordinates("poloidal")
        flux_surfaces.set_equilibrium(data[0].indica.equilibrium)
        binned_data = [bin_to_time_labels(times.data, d) for d in data]
        droppable_dims = [
            [dim for dim in d.dims if dim != d.attrs["transform"].x1_name] for d in data
        ]
        good_channels = [
            np.ravel(
                np.logical_not(
                    np.isnan(d.isel({dim: 0 for dim in droppable}))
                ).drop_vars(droppable)
            )
            for d, droppable in zip(data, droppable_dims)
        ]
        for d, g in zip(binned_data, good_channels):
            d.attrs["nchannels"] = (
                d.size
                * int(np.sum(g))
                // (d.coords[d.attrs["transform"].x1_name].size * times.size)
            )
        nt = len(times)
        rows = sum(d.attrs["nchannels"] for d in binned_data) * nt
        cols = (n_knots - 1) * nt
        sparsity = lil_matrix((rows, cols), dtype=int)
        nc1, nc2 = tee(d.attrs["nchannels"] for d in binned_data)

        for nc, data_row_start in zip(
            nc1, accumulate(map(lambda x: x * nt, chain(repeat(0, 1), nc2)))
        ):
            for i in range(nt):
                rstart = data_row_start + i * nc
                rend = rstart + nc
                cstart = i * (n_knots - 1)
                cend = cstart + (n_knots - 1)
                sparsity[rstart:rend, cstart:cend] = 1

        def knotvals_to_xarray(knotvals):
            all_knots = np.empty((nt, n_knots))
            all_knots[:, :-1] = knotvals.reshape(nt, n_knots - 1)
            all_knots[:, -1] = 0.0
            return DataArray(
                all_knots, coords=[("t", times), ("rho_poloidal", self.knots)]
            )

        # TODO: Consider how to handle locations outside of interpolation range.
        # For now just setting the interpolated values to 0.0
        def residuals(knotvals):
            self.spline_vals = knotvals_to_xarray(knotvals)
            self.spline = Spline(self.spline_vals, "rho_poloidal", flux_surfaces)
            start = 0
            resid = np.empty(rows)
            for d, g in zip(binned_data, good_channels):
                end = start + d.attrs["nchannels"] * nt
                rho, theta = d.indica.convert_coords(flux_surfaces)
                resid[start:end] = np.ravel(
                    (
                        self.spline(flux_surfaces, rho, theta, times).fillna(0.0) - d
                    ).isel({d.attrs["transform"].x1_name: g})
                )
                start = end
            # assert np.all(np.isfinite(resid))
            return resid

        guess = np.concatenate(
            tuple(
                np.mean([d.sel(t=t).mean() for d in binned_data]) * np.ones(n_knots - 1)
                for t in times
            )
        )

        fit = least_squares(
            residuals,
            guess,
            bounds=(self.lower_bound, self.upper_bound),
            jac_sparsity=sparsity,
            verbose=2,
        )

        if fit.status == -1:
            raise RuntimeError(
                "Improper input to `least_squares` function when trying to "
                "fit emissivity to radiation data."
            )
        elif fit.status == 0:
            warnings.warn(
                "Attempt to fit splines reached maximum number of function "
                "evaluations.",
                RuntimeWarning,
            )
        result = self.spline(flux_surfaces, rho, DataArray(0.0), times)
        result.attrs["splines"] = self.spline
        result.attrs["datatype"] = data[0].attrs["datatype"]
        # result.attrs["provenance"] = self.create_provenance()
        return result, self.spline_vals, *binned_data
