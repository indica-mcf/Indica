"""Various miscellanious helper functions."""

import string

import numpy as np


def sum_squares(x: np.ndarray, axis: int, **kwargs):
    """Perform a reduction on the provided data by taking the sum of squares.

    Parameters
    ----------
    x
        The arrayto be reduced.
    axis
        The axis along which to perform the reduction.
    \*\*kwargs
        Additiona keyword arguments (unused)

    """
    return np.sum(x**2, axis=axis)


def to_filename(name):
    """Takes a string and returns a valid filename based on it."""
    valid_chars = "-_.() {}{}".format(string.ascii_letters, string.digits)
    filename = name.replace("/", "-").replace("\\", "-")
    filename = filename.replace(" ", "_")
    filename = "".join(c for c in filename if c in valid_chars)
    return filename
