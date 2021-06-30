"""Defines an ArrayLike type, using either the new features available
to NumPy (still bleading edge) or a rough approximation. Also provides
a LabeledArray type, which corresponds to Xarray objects with
labeled dimensions and to scalar.

"""

import typing

import xarray

try:
    from numpy.typing import ArrayLike
except ImportError:
    import numpy as np
    import numbers

    ArrayLike = typing.Union[
        numbers.Number, typing.Sequence[numbers.Number], np.ndarray
    ]

LabeledArray = typing.Union[
    float, int, xarray.DataArray, xarray.Dataset, xarray.Variable
]

OnlyArray = typing.Union[
    typing.Sequence[numbers.Number],
    np.ndarray,
    xarray.DataArray,
    xarray.Dataset,
    xarray.Variable,
]

OnlyXarray = typing.Union[xarray.DataArray, xarray.Dataset, xarray.Variable]
