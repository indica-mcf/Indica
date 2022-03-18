"""Defines an ArrayLike type, using either the new features available
to NumPy (still bleading edge) or a rough approximation. Also provides
a LabeledArray type, which corresponds to Xarray objects with
labeled dimensions and to scalar.

"""

import numbers
import typing

import numpy as np
import xarray

# Jon added due to errors
import numpy as np
import numbers

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = typing.Union[
        numbers.Number, typing.Sequence[numbers.Number], np.ndarray
    ]

LabeledArray = typing.Union[
    float, int, np.ndarray, xarray.DataArray, xarray.Dataset, xarray.Variable
]

OnlyArray = typing.Union[
    typing.Sequence[numbers.Number],
    np.ndarray,
    xarray.DataArray,
    xarray.Dataset,
    xarray.Variable,
]

OnlyXarray = typing.Union[xarray.DataArray, xarray.Dataset, xarray.Variable]
