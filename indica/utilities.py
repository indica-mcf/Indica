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

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
from xarray import apply_ufunc
from xarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.variable import Variable

from .numpy_typing import ArrayLike
from .numpy_typing import LabeledArray
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
    positive: bool = True,
    strictly_positive: bool = True,
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
    positive
        Boolean, if true will check values >= 0
    strictly_positive
        Boolean, if true will check values > 0
    """
    if strictly_positive and not positive:
        raise ValueError("If checking for strictly_positive then set positive True.")

    if not isinstance(var_to_check, var_type):
        raise TypeError(f"{var_name} must be of type {var_type}.")

    # For some reason passing get_args(LabeledArray) to isinstance causes
    # mypy to complain but giving it the constituent types(and np.ndarray) solves this.
    # Guessing this is because LabeledArray isn't resolved/evaluated by mypy.
    # Return if not a numeric type, no additional checks required
    if not isinstance(
        var_to_check, (float, int, DataArray, Dataset, Variable, np.ndarray)
    ) or isinstance(var_to_check, bool):
        return

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

    if np.any(np.isnan(sliced_var_to_check)):
        raise ValueError(f"{var_name} cannot contain any NaNs.")

    if np.any(np.isinf(sliced_var_to_check)):
        raise ValueError(f"{var_name} cannot contain any infinities.")

    if positive and strictly_positive:
        if not np.all(sliced_var_to_check > 0):
            raise ValueError(f"Cannot have any negative or zero values in {var_name}")
    elif positive:
        if not np.all(sliced_var_to_check >= 0):
            raise ValueError(f"Cannot have any negative values in {var_name}")

    if (
        ndim_to_check is not None
        and isinstance(sliced_var_to_check, (np.ndarray, DataArray))
        and (sliced_var_to_check.ndim != ndim_to_check)
    ):
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


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0] - 1
    n2 = x2.shape[0] - 1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """
    INTERSECTIONS Intersections of curves.
       Computes the (x,y) locations where two curves intersect.  The curves
       can be broken with NaNs or have vertical segments.
    usage:
    x,y=intersection(x1,y1,x2,y2)
        Example:
        a, b = 1, 2
        phi = np.linspace(3, 10, 100)
        x1 = a*phi - b*np.sin(phi)
        y1 = a - b*np.cos(phi)
        x2=phi
        y2=np.sin(phi)+2
        x,y, ix, iy=intersection(x1,y1,x2,y2)
        plt.plot(x1,y1,c='r')
        plt.plot(x2,y2,c='g')
        plt.plot(x,y,'*k')
        plt.show()
    """
    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except np.linalg.LinAlgError:
            T[:, i] = np.NaN

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T

    indii = ii[in_range] + T[0, in_range]
    indjj = jj[in_range] + T[1, in_range]

    return xy0[:, 0], xy0[:, 1], indii, indjj


def check_time_present(t_desired: LabeledArray, t_array: LabeledArray):
    equil_ok = (np.min(t_desired) >= np.min(t_array)) * (
        np.max(t_desired) <= np.max(t_array)
    )
    if not equil_ok:
        raise ValueError(f"Desired time {t_desired} not available in array {t_array}")


def save_figure(
    path_name: str = "",
    fig_name: str = "",
    orientation: str = "landscape",
    dpi: int = 300,
    quality: int = 95,
    ext: str = "png",
    save_fig: bool = True,
):
    _fig_name = deepcopy(fig_name)
    _path_name = deepcopy(path_name)
    if _path_name[-1] != "/":
        _path_name = f"{_path_name}/"
    _file = f"{_path_name}{_fig_name}.{ext}"

    kwargs = {"orientation": orientation, "dpi": dpi}
    if ext != "svg":
        kwargs["pil_kwargs"] = {"quality": quality}

    if save_fig:
        plt.savefig(
            _file,
            **kwargs,
        )
        print(f"Saving picture to {_file}")
