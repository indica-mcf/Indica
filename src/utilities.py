"""Various miscellanious helper functions."""

import inspect
import string
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np


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


def sum_squares(x: np.ArrayLike, axis: int, **kwargs: Any) -> np.ArrayLike:
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


def get_slice_limits(low: float, high: float, data: np.ArrayLike) -> Tuple[int, int]:
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
