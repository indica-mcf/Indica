"""Various miscellanious helper functions."""

from copy import deepcopy
import inspect
import string
from typing import Any
from typing import Callable
from typing import Dict
from typing import Hashable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from scipy.interpolate import CubicSpline
from xarray import apply_ufunc
from xarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.variable import Variable

from indica.numpy_typing import LabeledArray
from .numpy_typing import ArrayLike
from .numpy_typing import OnlyArray


def positional_parameters(func: Callable[..., Any]) -> Tuple[List[str], Optional[str]]:
    """Returns an ordered list of the names of arguments which can be
    passed to a function positionally.

    Parameters
    ----------
    func
        A function/callable for which to get information on the positional
        arguments.

    Returns
    -------
    param_names : List[str]
        Ordered list of names of function parameters which can be passed to
        positionally.
    var_positional : str or None
        Name of variable positional parameter, if present (e.g., ``*args``).

    """
    param_names = []
    var_positional = None
    for param in inspect.signature(func).parameters.values():
        if (
            param.kind == inspect.Parameter.POSITIONAL_ONLY
            or param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ):
            param_names.append(param.name)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            var_positional = param.name
        else:
            break
    return param_names, var_positional


def sum_squares(x: OnlyArray, axis: int, **kwargs: Any) -> OnlyArray:
    """Perform a reduction on the provided data by taking the sum of squares.

    Parameters
    ----------
    x
        The arrayto be reduced.
    axis
        The axis along which to perform the reduction.
    **kwargs
        Additiona keyword arguments (unused)

    """
    return np.sum(x**2, axis=axis)


def get_slice_limits(low: float, high: float, data: OnlyArray) -> Tuple[int, int]:
    """Returns the start and end points needed to slice ``data`` so that
    all values fall between ``high`` and ``low`` (inclusive).

    An error will be raised if ``data`` does not contain values above and below
    these limits.

    Parameters
    ----------
    low
        The lower limit for the data.
    high
        The upper limit for the data.
    data
        An ordered 1-D array of values.

    Returns
    -------
    start
        The index above which all values of ``data`` are greater or equal to
        ``low``.
    end
        The index below which all values of ``data`` are less than or equal to
        ``high"

    """
    start = np.argmax(data > low) - 1
    if start < 0:
        raise ValueError("Low value {} not in range of provided " "data.".format(low))
    end = np.argmax(data >= high)
    if end < 1:
        raise ValueError("High value {} not in range of provided " "data.".format(high))

    return (start, end)  # type: ignore


def to_filename(name: str) -> str:
    """Takes a string and returns a valid filename based on it."""
    valid_chars = "-_.(){}{}".format(string.ascii_letters, string.digits)
    filename = name.replace("/", "-").replace("\\", "-")
    filename = filename.replace(" ", "_")
    filename = "".join(c for c in filename if c in valid_chars)
    return filename


def coord_array(coord_vals: ArrayLike, coord_name: str):
    """Create a DataArray which can be used for a coordinate system.

    Parameters
    ----------
    coord_vals
        The position/value at each point on the coordinate grid.
    coord_name
        The name of the dimension.
    """
    datatype = (
        "major_rad"
        if coord_name == "R"
        else "time"
        if coord_name == "t"
        else "norm_flux_pol"
        if coord_name == "rho_poloidal"
        else "norm_flux_tor"
        if coord_name == "rho_toroidal"
        else coord_name,
        "plasma",
    )
    return DataArray(
        coord_vals, coords=[(coord_name, coord_vals)], attrs={"datatype": datatype}
    )


def broadcast_spline(
    spline: CubicSpline,
    spline_dims: Tuple,
    spline_coords: Dict[Hashable, Any],
    interp_coord: DataArray,
):
    """Applies to a `:class:xarray.DataArray` input, broadcasting and/or
    interpolating appropriately.

    Note
    ----
    Currently the only dimension which are checked for presence in
    both the spline data and the input is "t".

    TODO: Implement these checks for other dimensions as well.

    """
    if "t" in interp_coord.coords and "t" in spline_dims:
        time_outer_product = apply_ufunc(
            spline,
            interp_coord,
            input_core_dims=[[]],
            output_core_dims=[tuple(d if d != "t" else "__old_t" for d in spline_dims)],
        ).assign_coords(__old_t=coord_array(spline_coords["t"].data, "__old_t"))
        result = time_outer_product.indica.interp2d(
            __old_t=interp_coord.coords["t"],
            method="cubic"
            if time_outer_product.coords["__old_t"].size > 3
            else "linear",
        ).assign_coords({k: v for k, v in spline_coords.items() if k != "t"})
        del result.coords["__old_t"]
        return result
    else:
        return apply_ufunc(
            spline,
            interp_coord,
            input_core_dims=[[]],
            output_core_dims=[spline_dims],
        ).assign_coords({k: v for k, v in spline_coords.items()})


def input_check(
    var_name: str,
    var_to_check,
    var_type: Union[type, Tuple[type, ...]],
    ndim_to_check: Optional[int] = None,
    greater_than_or_equal_zero: Optional[bool] = False,
):
    """Check validity of inputted variable - type check and
    various value checks(no infinities, greather than (or equal to) 0 or NaNs)

    Parameters
    ----------
    var_name
        Name of variable to check.
    var_to_check
        Variable to check.
    var_type
        Type to check variable against, eg. DataArray
    ndim_to_check
            Integer to check the number of dimensions of the variable.
    greater_than_or_equal_zero
        Boolean to check values in variable > 0 or >= 0.
    """

    try:
        assert isinstance(var_to_check, var_type)
    except AssertionError:
        raise TypeError(f"{var_name} must be of type {var_type}.")

    # For some reason passing get_args(LabeledArray) to isinstance causes
    # mypy to complain but giving it the constituent types(and np.ndarray) solves this.
    # Guessing this is because LabeledArray isn't resolved/evaluated by mypy.
    if isinstance(var_to_check, (float, int, DataArray, Dataset, Variable, np.ndarray)):

        # Handles dropped channels, if present
        sliced_var_to_check = deepcopy(var_to_check)
        if (
            isinstance(var_to_check, (DataArray, Dataset))
            and "dropped" in var_to_check.attrs
        ):
            dropped_coords = var_to_check.attrs["dropped"].coords
            for icoord in dropped_coords.keys():
                dropped_coord = dropped_coords[icoord]
                sliced_var_to_check = var_to_check.drop_sel({icoord: dropped_coord})

        try:
            assert np.all(np.logical_not(np.isnan(sliced_var_to_check)))
        except AssertionError:
            raise ValueError(f"{var_name} cannot contain any NaNs.")

        try:
            assert np.all(np.logical_not(np.isinf(np.abs(sliced_var_to_check))))
        except AssertionError:
            raise ValueError(f"{var_name} cannot contain any infinities.")

        if not greater_than_or_equal_zero:
            try:
                assert np.all(sliced_var_to_check > 0)
            except AssertionError:
                raise ValueError(
                    f"Cannot have any negative or zero values in {var_name}"
                )
        else:
            try:
                assert np.all(sliced_var_to_check >= 0)
            except AssertionError:
                raise ValueError(f"Cannot have any negative values in {var_name}")

    if ndim_to_check is not None and isinstance(
        sliced_var_to_check, (np.ndarray, DataArray)
    ):
        try:
            assert sliced_var_to_check.ndim == ndim_to_check
        except AssertionError:
            raise ValueError(f"{var_name} must have {ndim_to_check} dimensions.")


def assign_datatype(data_array: DataArray, datatype: tuple, unit=""):
    data_array.name = f"{datatype[1]}_{datatype[0]}"
    data_array.attrs["datatype"] = datatype
    if len(unit) > 0:
        data_array.attrs["unit"] = unit


def assign_data(
    data: LabeledArray,
    datatype: tuple,
    units: str = "",
    make_copy=True,
    coords: list = None,
    long_name: str = None,
):
    new_data: DataArray

    if make_copy:
        new_data = deepcopy(data)
    else:
        new_data = data

    if type(new_data) is not DataArray and coords is not None:
        new_data = DataArray(new_data, coords)

    new_data.name = f"{datatype[1]}_{datatype[0]}"
    datatype0 = datatype[0].replace("_", " ")
    datatype1 = datatype[1].replace("_", " ")
    datatype1 = f"{str.upper(datatype1[0])}{datatype1[1:]}"

    if long_name is not None:
        new_data.attrs["long_name"] = long_name
    else:
        new_data.attrs["long_name"] = f"{datatype1} {datatype0}"

    new_data.attrs["datatype"] = datatype

    if len(units) > 0:
        new_data.attrs["units"] = units

    return new_data


def print_like(string: str):
    print(f"\n {string}")


def get_function_name():
    return str(inspect.stack()[1][3])
