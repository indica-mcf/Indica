"""Various miscellanious helper functions."""

import inspect
import string
from typing import Any
from typing import Callable
from typing import Dict
from typing import Hashable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from xarray import apply_ufunc
from xarray import DataArray

from .numpy_typing import ArrayLike


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


def sum_squares(x: ArrayLike, axis: int, **kwargs: Any) -> ArrayLike:
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
    return np.sum(x ** 2, axis=axis)


def get_slice_limits(low: float, high: float, data: ArrayLike) -> Tuple[int, int]:
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

    return (start, end)


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
            __old_t=interp_coord.coords["t"], method="cubic"
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
