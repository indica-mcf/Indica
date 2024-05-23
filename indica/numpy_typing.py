"""Defines an ArrayLike type, using either the new features available
to NumPy (still bleading edge) or a rough approximation. Also provides
a LabeledArray type, which corresponds to Xarray objects with
labeled dimensions and to scalar.

"""

import numbers
import typing
from typing import Optional
from typing import Tuple

import numpy as np
import xarray

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = typing.Union[  # type: ignore
        numbers.Number, typing.Sequence[numbers.Number], np.ndarray
    ]

LabeledArray = typing.Union[
    float, int, np.ndarray, xarray.DataArray, xarray.Dataset, xarray.Variable, list
]

OnlyArray = typing.Union[
    np.ndarray,
    xarray.DataArray,
    xarray.Dataset,
    xarray.Variable,
]

OnlyXarray = typing.Union[xarray.DataArray, xarray.Dataset, xarray.Variable]

# Type for generic functions that e.g. return same type as input
XarrayGeneric = typing.TypeVar("XarrayGeneric", xarray.DataArray, xarray.Dataset)

RevisionLike = typing.Union[str, int]

Coordinates = Tuple[LabeledArray, LabeledArray]

OptionalCoordinates = Tuple[Optional[LabeledArray], Optional[LabeledArray]]

FloatOrDataArray = typing.Union[xarray.DataArray, float]