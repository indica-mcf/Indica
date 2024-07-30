"""
`"Custom accessors"
<https://xarray.pydata.org/en/stable/internals/extending-xarray.html>`_,
are defined to provide additional functionality to
:py:class:`xarray.DataArray` and :py:class:`xarray.Dataset`
objects. These accessors group methods under the namespace
``indica``. E.g.::

"""

from numbers import Number
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Hashable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import RectBivariateSpline
import xarray as xr
from xarray.core.utils import either_dict_or_kwargs

from .converters import CoordinateTransform
from .converters.abstractconverter import Coordinates
from .equilibrium import Equilibrium
from .numpy_typing import ArrayLike
from .numpy_typing import LabeledArray


def _convert_coords(
    array: Union[xr.Dataset, xr.DataArray], transform: CoordinateTransform
) -> Coordinates:
    """Convert this array's coordinates to those of the coordinate system
    ``transform``. The result will be cached for future reuse.

    Parameters
    ----------
    array
        The object for which coordinates will be converted.
    transform
        The transform describing the coordinate system to which the
        coordiantes will be converted.

    Returns
    -------
    x1 : LabeledArray
        The first spatial coordinate in the new system
    x2 : LabeledArray
        The second spatial coordinate in the new system

    """
    if transform.x1_name not in array.coords or transform.x2_name not in array.coords:
        self_trans: CoordinateTransform = array.attrs["transform"]
        converter = self_trans.get_converter(transform)
        if converter:
            x1, x2 = converter(
                array.coords[self_trans.x1_name],
                array.coords[self_trans.x2_name],
                array.coords["t"],
            )
        else:
            if "R" not in array.coords or "z" not in array.coords:
                R, z = self_trans.convert_to_Rz(
                    array.coords[self_trans.x1_name],
                    array.coords[self_trans.x2_name],
                    array.coords["t"],
                )
                if "R" not in array.coords:
                    array.coords["R"] = R
                if "z" not in array.coords:
                    array.coords["z"] = z
            else:
                R = array.coords["R"]
                z = array.coords["z"]
            x1, x2 = transform.convert_from_Rz(R, z, array.coords["t"])
        if transform.x1_name not in array.coords:
            array.coords[transform.x1_name] = x1
        if transform.x2_name not in array.coords:
            array.coords[transform.x2_name] = x2
    return array.coords[transform.x1_name], array.coords[transform.x2_name]


@xr.register_dataarray_accessor("indica")
class InDiCAArrayAccessor:
    """Class providing additional functionality to
    :py:class:`xarray.DataArray` objects which is useful for this software.

    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def invert_interp(
        self,
        values: ArrayLike,
        target: str,
        new_dims: Optional[Union[Tuple[str, ...], str]] = None,
        coords: Optional[Mapping[Hashable, ArrayLike]] = None,
        method: str = "linear",
        assume_sorted: bool = False,
        kwargs: Optional[Mapping[str, Any]] = None,
        **coords_kwargs: ArrayLike,
    ) -> xr.DataArray:
        """Performs an inversion to give coordinates at which the data has the
        specified value. Interpolation will be performed with the
        specified method. This approach is efficient if you need to do
        the inversion on a large number of values, but **will only
        work properly if the data is monotonic on the target
        dimension**.

        Parameters
        ----------
        values
            The values of data at which to invert.
        target
            The name of the dimension on which the result contains the
            coordinates at which ``values`` occur.
        new_dims
            The name(s) of the new dimensions introduced by inverting with
            ``values``. Defaults to the name of the original DataArray. If that
            array does not have a name and a named dimension in ``values`` is
            equal to ``target`` then this argument is required.
        coords
            A mapping from dimension names to new coordinates onto
            which to interpolate data. New coordinate can be a scalar,
            array-like or DataArray. If DataArrays are passed as new
            coordinates, their dimensions are used for the
            broadcasting. Missing values are skipped. The target
            dimension **must not be present.**
        method
            {“linear”, “nearest”} for multidimensional array,
            {“linear”, “nearest”, “zero”, “slinear”, “quadratic”,
            “cubic”} for 1-dimensional array. “linear” is used by
            default.
        assume_sorted
            If False, values of coordinates that are interpolated over
            can be in any order and they are sorted first. If True,
            interpolated coordinates are assumed to be an array of
            monotonically increasing values.
        kwargs
            Additional keyword arguments passed to scipy’s
            interpolator. Valid options and their behavior depend on
            if 1-dimensional or multi-dimensional interpolation is
            used.
        coords_kwargs
            The keyword arguments form of coords.

        Returns
        -------
        :
            A DataArray containing the positions on the ``target`` dimension at
            which the original DataArray has ``values``, interpolated as
            necessary.

        """

        def invert_interp_func(data, target_coords, vals):
            not_nan = np.logical_not(np.isnan(data))
            interp = interp1d(data[not_nan], target_coords[not_nan], method, copy=False)
            return interp(vals)

        if coords or coords_kwargs:
            if (coords and target in coords) or target in coords_kwargs:
                raise ValueError(
                    f"Target dimension ('{target}') must not be present in coords."
                )
            interpolated = self._obj.interp(
                coords, method, assume_sorted, kwargs, **coords_kwargs
            )
        else:
            interpolated = self._obj
        if not isinstance(values, xr.DataArray):
            values = xr.DataArray(values)
        new_dim_names: List[str]
        if new_dims is None:
            if values.ndim == 0:
                new_dim_names = []
                value_dims = new_dim_names
            elif target in values.dims:
                if self._obj.name is None:
                    raise ValueError("Values must have a name if 'new_dims' is None.")
                new_dim_names = [
                    cast(str, self._obj.name),
                ]
                value_dims = [
                    target,
                ]
            else:
                new_dim_names = []
                value_dims = new_dim_names
        elif isinstance(new_dims, str):
            new_dim_names = [
                new_dims,
            ]
            value_dims = cast(List, values.dims)
        else:
            new_dim_names = list(new_dims)
            value_dims = cast(List, values.dims)
        data = xr.apply_ufunc(
            invert_interp_func,
            interpolated,
            interpolated.coords[target],
            values,
            input_core_dims=[[target], [target], value_dims],
            output_core_dims=[new_dim_names],
            exclude_dims=set(value_dims),
            vectorize=True,
        )
        data.name = target
        return data

    def invert_root(
        self,
        values: ArrayLike,
        target: str,
        guess: Optional[ArrayLike] = None,
        new_dims: Optional[Union[Tuple[str, ...], str]] = None,
        coords: Optional[Mapping[Hashable, ArrayLike]] = None,
        method: str = "linear",
        assume_sorted: bool = False,
        kwargs: Optional[Mapping[str, Any]] = None,
        **coords_kwargs: ArrayLike,
    ) -> xr.DataArray:
        """Performs an inversion to give coordinates at which the data has the
        specified value. Cubic spline interpolation will be performed with the
        specified method and then the root(s) of the interpolant will
        be found. If there are multiple roots then a guess must be
        provided to determine which one to use.

        Parameters
        ----------
        values
            The values of data at which to invert.
        target
            The name of the dimension on which the result contains the
            coordinates at which ``values`` occur.
        guess
            An estimate of the values being solved for. If there are multiple
            roots then the one closest to the guess will be used.
        new_dims
            The name(s) of the new dimensions introduced by inverting with
            ``values``. Defaults to the name of the original DataArray. If that
            array does not have a name and a named dimension in ``values`` is
            equal to ``target`` then this argument is required.
        coords
            A mapping from dimension names to new coordinates onto
            which to interpolate data. New coordinate can be a scalar,
            array-like or DataArray. If DataArrays are passed as new
            coordinates, their dimensions are used for the
            broadcasting. Missing values are skipped. The target
            dimension **must not be present.**
        method
            {“linear”, “nearest”} for multidimensional array,
            {“linear”, “nearest”, “zero”, “slinear”, “quadratic”,
            “cubic”} for 1-dimensional array. “linear” is used by
            default. Only applies for interpolation onto ``coords``.
        assume_sorted
            If False, values of coordinates that are interpolated over
            can be in any order and they are sorted first. If True,
            interpolated coordinates are assumed to be an array of
            monotonically increasing values.
        kwargs
            Additional keyword arguments passed to scipy’s
            interpolator. Valid options and their behavior depend on
            if 1-dimensional or multi-dimensional interpolation is
            used.
        coords_kwargs
            The keyword arguments form of coords.

        Returns
        -------
        :
            A DataArray containing the positions on the ``target`` dimension at
            which the original DataArray has ``values``, interpolated as
            necessary.

        """

        def invert_root_func(data, target_coords, value, local_guess):
            not_nan = np.logical_not(np.isnan(data))
            interp = InterpolatedUnivariateSpline(
                target_coords[not_nan], (data - value)[not_nan], ext=2
            )
            roots = interp.roots()
            if len(roots) == 0:
                start = target_coords.argmin()
                end = target_coords.argmax()
                if np.isclose(data[start], value):
                    roots = np.concatenate((roots, [target_coords[start]]))
                if np.isclose(data[end], value):
                    roots = np.concatenate((roots, [target_coords[end]]))
                if len(roots) == 0:
                    raise ValueError(f"Provided data is not available at {value}.")
            elif len(roots) > 1 and guess is None:
                raise ValueError(
                    "A guess must be provided when there is more than one root."
                )
            i = np.abs(roots - local_guess).argmin()
            return roots[i]

        if coords or coords_kwargs:
            if (coords and target in coords) or target in coords_kwargs:
                raise ValueError(
                    f"Target dimension ('{target}') must not be present in coords."
                )
            interpolated = self._obj.interp(
                coords, method, assume_sorted, kwargs, **coords_kwargs
            )
        else:
            interpolated = self._obj
        if not isinstance(values, xr.DataArray):
            values = xr.DataArray(values)
        new_dim_names: List[str]
        value_dims: List[str]
        if new_dims is None:
            if values.ndim == 0:
                new_dim_names = []
                value_dims = new_dim_names
            elif target in values.dims:
                if values.ndim == 1:
                    if self._obj.name is None:
                        raise ValueError(
                            "Values must have a name if 'new_dims' is None."
                        )
                    new_dim_names = [
                        cast(str, self._obj.name),
                    ]
                    value_dims = [
                        target,
                    ]
                else:
                    raise ValueError(
                        f"Target dimension '{target}' can only be used in `values` "
                        "if `values` is a 1-D array."
                    )
            else:
                new_dim_names = []
                value_dims = new_dim_names
        elif isinstance(new_dims, str):
            new_dim_names = [
                new_dims,
            ]
        else:
            new_dim_names = list(new_dims)

        # FIXME: Still need to handling of guess dimensions...
        if isinstance(guess, xr.DataArray) and target in guess.dims:
            guess_core_dims = [target]
        else:
            guess_core_dims = []
        if isinstance(values, xr.DataArray):
            value_dims = [cast(str, dim) for dim in values.dims if dim in new_dim_names]
        else:
            value_dims = []
        data = xr.apply_ufunc(
            invert_root_func,
            interpolated,
            interpolated.coords[target],
            values,
            0.0 if guess is None else guess,
            input_core_dims=[[target], [target], value_dims, guess_core_dims],
            output_core_dims=[new_dim_names],
            exclude_dims=set((target,)),
            vectorize=True,
        )
        data.name = target
        return data

    @staticmethod
    def _get_unlabeled_interpolation_1d(
        assume_sorted: bool, degree: int
    ) -> Callable[[ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]], ArrayLike]:
        def interpolate_1d(x, z, x_interp, x_zero):
            if assume_sorted and x_zero is None:
                interpolant = InterpolatedUnivariateSpline(x, z, k=degree)
                domain = (x[0], x[-1])
            else:
                if x_zero is not None and x_zero not in x:
                    x = np.append(x, x_zero)
                    z = np.append(z, 0.0)
                order = np.argsort(x)
                interpolant = InterpolatedUnivariateSpline(x[order], z[order], k=degree)
                domain = (x[order[0]], x[order[-1]])
            result = interpolant(x_interp)
            result[np.logical_or(x_interp < domain[0], x_interp > domain[1])] = float(
                "nan"
            )
            return result

        return interpolate_1d

    @staticmethod
    def _get_unlabeled_interpolation_2d(
        assume_sorted: bool, degree: int
    ) -> Callable[
        [
            ArrayLike,
            ArrayLike,
            ArrayLike,
            ArrayLike,
            ArrayLike,
            Optional[ArrayLike],
            Optional[ArrayLike],
        ],
        ArrayLike,
    ]:
        def interpolate_2d(x, y, z, x_interp, y_interp, x_zero, y_zero):
            if assume_sorted:
                interpolant = RectBivariateSpline(x, y, z, kx=degree, ky=degree)
                result = interpolant(x_interp, y_interp)
                domain = ((x[0], x[-1]), (y[0], y[-1]))
            else:
                xorder = np.argsort(x)
                yorder = np.argsort(y)
                z_ordered = np.take(np.take(z, xorder, 0), yorder, 1)
                interpolant = RectBivariateSpline(
                    np.take(x, xorder),
                    np.take(y, yorder),
                    z_ordered,
                    kx=degree,
                    ky=degree,
                )
                new_xorder = np.argsort(x_interp)
                new_yorder = np.argsort(y_interp)
                vals = interpolant(
                    np.take(x_interp, new_xorder),
                    np.take(y_interp, new_yorder),
                ).T
                vals2 = np.empty_like(vals)
                result = np.empty_like(vals)
                np.put(vals2, new_xorder, vals, 1)
                np.put(result, new_yorder, vals2, 0)
                domain = ((x[xorder[0]], x[xorder[-1]]), (y[yorder[0]], y[yorder[-1]]))
            if (
                x_zero is not None
                and y_zero is not None
                and (x_zero not in x or y_zero not in y)
            ):
                x_offsets = x - x_zero
                y_offsets = y - y_zero
                x_below_zero_i = np.argmax(
                    np.where(x_offsets < 0, x_offsets, float("-inf"))
                )
                x_above_zero_i = np.argmin(
                    np.where(x_offsets > 0, x_offsets, float("inf"))
                )
                y_below_zero_i = np.argmax(
                    np.where(y_offsets < 0, y_offsets, float("-inf"))
                )
                y_above_zero_i = np.argmin(
                    np.where(y_offsets > 0, y_offsets, float("inf"))
                )
                x_below_zero = x[x_below_zero_i]
                x_above_zero = x[x_above_zero_i]
                y_below_zero = y[y_below_zero_i]
                y_above_zero = y[y_above_zero_i]
                xmask = np.logical_and(
                    x_interp >= x_below_zero, x_interp <= x_above_zero
                )
                ymask = np.logical_and(
                    y_interp >= y_below_zero, y_interp <= y_above_zero
                )
                if isinstance(xmask, np.ndarray) and isinstance(ymask, np.ndarray):
                    xmask = np.expand_dims(xmask, 1)
                    x_interp = np.expand_dims(x_interp, 1)
                mask = np.logical_and(xmask, ymask)
                if np.any(mask):
                    interpolant = CloughTocher2DInterpolator(
                        np.array(
                            [
                                [x_below_zero, y_below_zero],
                                [x_below_zero, y_above_zero],
                                [x_above_zero, y_below_zero],
                                [x_above_zero, y_above_zero],
                                [x_zero, y_zero],
                            ]
                        ),
                        np.array(
                            [
                                z[x_below_zero_i, y_below_zero_i],
                                z[x_below_zero_i, y_above_zero_i],
                                z[x_above_zero_i, y_below_zero_i],
                                z[x_above_zero_i, y_above_zero_i],
                                0.0,
                            ]
                        ),
                    )
                    xx_interp, yy_interp = np.broadcast_arrays(x_interp, y_interp)
                    result[mask] = interpolant(
                        np.stack([xx_interp[mask], yy_interp[mask]], axis=-1)
                    )
            out_of_domain = np.logical_or(
                np.logical_or(x_interp < domain[0][0], x_interp > domain[0][1]),
                np.logical_or(y_interp < domain[1][0], y_interp > domain[1][1]),
            )
            result[out_of_domain] = float("nan")
            return result

        return interpolate_2d

    def interp2d(
        self,
        coords: Optional[Mapping[Hashable, ArrayLike]] = None,
        zero_coords: Optional[Mapping[Hashable, ArrayLike]] = None,
        method: Union[str, int] = "linear",
        assume_sorted: bool = False,
        **coords_kwargs: ArrayLike,
    ) -> xr.DataArray:
        """Performs interpolation along two dimensions simultatiously. Unlike
        :py:meth:`xarray.DataArray.interp`, this routine supports
        higher-order interpolation than linear by usinmg
        :py:func:`scipy.interpolate.RectBivariateSpline` (although
        with potentially-inefficient vectorisation). All options are the same
        as in the xarray method. However, interpolation will not be performed
        on any of the non-dimensional coordinates, unlike in the xarray method.

        Parameters
        ----------
        coords
            A mapping from dimension names to new coordinates onto
            which to interpolate data. New coordinate can be a scalar,
            array-like or DataArray. If DataArrays are passed as new
            coordinates, their dimensions are used for the
            broadcasting. Missing values are skipped.
        zero_coords
            Coordinates for any points where the data is known to be equal to
            zero but which are not captured by the grid.
        method
            A string {“linear”, “nearest”, “quadratic”, “cubic”} or an integer
            indicating the degree of the splines.
        assume_sorted
            If False, values of coordinates that are interpolated over
            can be in any order and they are sorted first. If True,
            interpolated coordinates are assumed to be an array of
            monotonically increasing values.
        coords_kwargs
            The keyword arguments form of coords.

        Returns
        -------
        :
            A DataArray containing the positions on the ``target`` dimension at
            which the original DataArray has ``values``, interpolated as
            necessary.

        """
        METHODS = {"nearest": 0, "linear": 1, "quadratic": 2, "cubic": 3}
        if isinstance(method, str):
            degree = METHODS[method]
        else:
            degree = method

        _coords = either_dict_or_kwargs(coords, coords_kwargs, "indica.interp2d")
        if len(_coords) > 2:
            raise ValueError("Must not pass more than 2 coordinates to interp2d.")
        if zero_coords and len(zero_coords) != len(_coords):
            raise ValueError(
                "Argument zero_coords must contain the same number of entries "
                "as coordinates on which interpolation is being performed."
            )
        ordered_coords: List[Tuple[str, Union[xr.DataArray, Number]]] = []
        output_core: List[str] = []
        interp_core: List[List[str]] = []
        rename_dims: Dict[str, str] = {}
        ordered_zero_coords: List[ArrayLike] = []
        for k, v in _coords.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                new_dim = "__new_" + cast(str, k)
                ordered_coords.append((cast(str, k), xr.DataArray(v, dims=new_dim)))
                output_core.append(new_dim)
                interp_core.append([new_dim])
                rename_dims[new_dim] = cast(str, k)
            elif isinstance(v, xr.DataArray):
                if k in v.dims:
                    new_dim = "__new_" + cast(str, k)
                    interp_core.append([new_dim])
                    rename_dims[new_dim] = cast(str, k)
                    if new_dim not in output_core:
                        output_core.append(new_dim)
                    ordered_coords.append((cast(str, k), v.rename({k: new_dim})))
                else:
                    interp_core.append([])
                    ordered_coords.append((cast(str, k), v))
            else:
                # Not really sure what mypy is complaining about here.
                ordered_coords.append((k, v))  # type: ignore
                # mypy doesn't allow empty append.
                output_core.append([])  # type: ignore
            if zero_coords:
                ordered_zero_coords.append(zero_coords[k])
            else:
                ordered_zero_coords.append(None)  # type: ignore
        if len(_coords) > 1:
            input_core: List[List[str]] = [
                [ordered_coords[0][0]],
                [ordered_coords[1][0]],
                list(cast(Mapping[str, Any], _coords)),
            ]
            result = xr.apply_ufunc(
                self._get_unlabeled_interpolation_2d(assume_sorted, degree),
                self._obj.coords[ordered_coords[0][0]],
                self._obj.coords[ordered_coords[1][0]],
                self._obj,
                ordered_coords[0][1],
                ordered_coords[1][1],
                ordered_zero_coords[0],
                ordered_zero_coords[1],
                input_core_dims=input_core
                + interp_core
                + cast(List[List[str]], [[], []]),
                output_core_dims=[output_core],
                exclude_dims=set(_coords),
                vectorize=True,
            )
        else:
            input_core = [
                [ordered_coords[0][0]],
                list(cast(Mapping[str, Any], _coords)),
            ]
            result = xr.apply_ufunc(
                self._get_unlabeled_interpolation_1d(assume_sorted, degree),
                self._obj.coords[ordered_coords[0][0]],
                self._obj,
                ordered_coords[0][1],
                ordered_zero_coords[0],
                input_core_dims=input_core + interp_core + cast(List[List[str]], [[]]),
                output_core_dims=[output_core],
                exclude_dims=set(_coords),
                vectorize=True,
            )
        if len(rename_dims) > 0:
            result = result.rename(rename_dims)
        for name, coord in ordered_coords:
            if name not in result.coords:
                result.coords[name] = coord
        return result

    def convert_coords(self, transform: CoordinateTransform) -> Coordinates:
        """Convert this array's coordinates to those of the coordinate system
        ``transform``. The result will be cached for future reuse.

        Parameters
        ----------
        transform
            The transform describing the coordinate system to which the
            coordiantes will be converted.

        Returns
        -------
        x1 : LabeledArray
            The first spatial coordinate in the new system
        x2 : LabeledArray
            The second spatial coordinate in the new system

        """
        return _convert_coords(self._obj, transform)

    def get_coords(
        self, transform: Optional[CoordinateTransform] = None
    ) -> Tuple[LabeledArray, LabeledArray, LabeledArray]:
        """Get this array's coordinates, including time, in the coordinate
        system ``transform``. The result will be cached for future
        reuse.

        Parameters
        ----------
        transform
            The transform describing the coordinate system to which the
            coordiantes will be converted.

        Returns
        -------
        x1 : LabeledArray
            The first spatial coordinate in the new system
        x2 : LabeledArray
            The second spatial coordinate in the new system
        t : LabeledArray
            The time coordinate

        """
        if transform is None:
            transform = self._obj.attrs["transform"]
        return self.convert_coords(transform) + (self._obj.coords["t"],)

    @property
    def equilibrium(self) -> Optional[Equilibrium]:
        """The equilibrium object currently used by this DataArray (or, more
        accurately, by its
        :py:class:`~indica.converters.CoordinateTransform`
        object).

        """
        if "transform" in self._obj.attrs:
            return getattr(self._obj.attrs["transform"], "equilibrium", None)
        else:
            return None

    @equilibrium.setter
    def equilibrium(self, value: Equilibrium):
        if value == self.equilibrium:
            return
        self._obj.attrs["transform"].set_equilibrium(value)

    @equilibrium.deleter
    def equilibrium(self):
        if hasattr(self._obj.attrs["transform"], "equilibrium"):
            del self._obj.attrs["transform"].equilibrium


@xr.register_dataset_accessor("indica")
class InDiCADatasetAccessor:
    """Class providing additional functionality to
    :py:class:`xarray.Dataset` objects which is useful for this software.

    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def convert_coords(self, transform: CoordinateTransform) -> Coordinates:
        """Convert this dataset's coordinates to those of the coordinate system
        ``transform``. The result will be cached for future reuse.

        Parameters
        ----------
        transform
            The transform describing the coordinate system to which the
            coordiantes will be converted.

        Returns
        -------
        x1 : LabeledArray
            The first spatial coordinate in the new system
        x2 : LabeledArray
            The second spatial coordinate in the new system

        """
        return _convert_coords(self._obj, transform)

    def get_coords(
        self, transform: Optional[CoordinateTransform] = None
    ) -> Tuple[LabeledArray, LabeledArray, LabeledArray]:
        """Get this dataset's coordinates, including time, in the coordinate
        system ``transform``. The result will be cached for future
        reuse.

        Parameters
        ----------
        transform
            The transform describing the coordinate system to which the
            coordiantes will be converted.

        Returns
        -------
        x1 : LabeledArray
            The first spatial coordinate in the new system
        x2 : LabeledArray
            The second spatial coordinate in the new system
        t : LabeledArray
            The time coordinate

        """
        if transform is None:
            transform = self._obj.attrs["transform"]
        return self.convert_coords(transform) + (self._obj.coords["t"],)
