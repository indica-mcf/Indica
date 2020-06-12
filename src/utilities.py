"""Various miscellanious helper functions."""

import inspect
import string

import numpy as np


def positional_parameters(func):
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
        if param.kind == inspect.Parameter.POSITIONAL_ONLY or \
           param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            param_names.append(param.name)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            var_positional = param.name
        else:
            break
    return param_names, var_positional


def sum_squares(x: np.ndarray, axis: int, **kwargs):
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


def to_filename(name):
    """Takes a string and returns a valid filename based on it."""
    valid_chars = "-_.(){}{}".format(string.ascii_letters, string.digits)
    filename = name.replace("/", "-").replace("\\", "-")
    filename = filename.replace(" ", "_")
    filename = "".join(c for c in filename if c in valid_chars)
    return filename
