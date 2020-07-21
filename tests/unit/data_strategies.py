"""Some strategies for generating artificial data on which computations can be
 performed.

"""

from unittest.mock import MagicMock

from hypothesis import composite
from hypothesis import integers
from hypothesis import just
from hypothesis import lists
from hypothesis import sampled_from
from hypothesis import text
import numpy as np
from xarray import DataArray
from xarray import Dataset

import src.datatypes as dt
from .converters.test_abstract_transform import coordinate_transforms
from .strategies import noisy_functions
from .strategies import separable_functions
from .strategies import smooth_functions


@composite
def general_datatypes(draw, specific_datatype=None):
    """A strategy to select one of the general data types defined in
    :py:mod:`src.datatypes`. If ``specific_datatype`` is present then the
    result will be compatible with that.

    """

    def valid_datatype(datatype):
        if specific_datatype:
            return datatype in dt.COMPATIBLE_DATATYPES[specific_datatype]
        return True

    if specific_datatype:
        return draw(sampled_from(dt.GENERAL_DATATYPES.keys()).filter(valid_datatype))


@composite
def specific_datatypes(draw, general_datatype=None):
    """A strategy to select one of the specific data types defined in
    :py:mod:`src.datatypes`. If ``general_datatype`` is present then the result
    will be compatible with that.

    """

    def valid_datatype(datatype):
        if general_datatype:
            return general_datatype in dt.COMPATIBLE_DATATYPES[datatype]
        return True

    return draw(sampled_from(dt.SPECIFIC_DATATYPES.keys()).filter(valid_datatype))


@composite
def dropped_channels(draw, size, max_dropped=0.1):
    """A strategy to generate a list of channels to drop, given the total
    number of channels.

    Parameters
    ----------
    size
        The number of channels
    max_dropped
        The maximum number of channels to drop, as a fraction of the total
        number of channels.

    """
    return draw(
        lists(integers(0, size - 1), max_size=int(max_dropped * size), unique=True)
    )


@composite
def data_arrays(
    draw,
    data_type=(None, None),
    coordinates=coordinate_transforms(((1.83, 3.9), (-1.75, 2.0), (50.0, 120.0)), 4, 3),
    data=separable_functions(
        smooth_functions((1.83, 3.9)),
        smooth_functions(-1.75, 2.0),
        smooth_functions(50.0, 120.0),
    ),
    rel_sigma=0.02,
    abs_sigma=-1e-3,
    uncertainty=True,
    max_dropped=0.1,
):
    """Returns a DataArray, with appropriate metadata for use in testing
    calculations.

    Parameters
    ----------
    data_type : Tuple[str, str]
        The data type of the data_array to be generated. If either element of
        the tuple is ``None`` then that element will be drawn from a strategy.
    coordinates
        A strategy for generating :py:class:`src.converters.CoordinateTransform`
        objects. If absent, any type of transform could be used.
    data
        A strategy to generate functions which calculate the contents of the
        DataArray from the coordinates.
    rel_sigma
        Standard deviation of relative noise applied to the data
    abs_sigma
        Standard deviation of absolute noise applied to the data
    uncertainty
        If ``True``, generate uncertainty metadata using ``rel_sigma`` and
        ``abs_sigma`` (if they are non-zero).
    max_dropped
        The maximum number of channels to drop, as a fraction of the total
        number of channels.

    """
    general_type = (
        data_type[0] if data_type[0] else draw(general_datatypes(data_type[1]))
    )
    specific_type = (
        data_type[1] if data_type[1] else draw(specific_datatypes(general_type))
    )
    transform = draw(coordinates)
    x1 = transform.default_x1
    x2 = transform.default_x2
    t = transform.default_t
    func = (
        draw(noisy_functions(draw(data), rel_sigma, abs_sigma))
        if rel_sigma or abs_sigma
        else draw(data)
    )
    coords = [
        c
        for c in [("x1", x1), ("x2", x2), ("t", t)]
        if isinstance(c[1], np.ndarray) and c[1].ndims > 0
    ]
    result = DataArray(func(x1, x2, t), coords=coords)
    dropped = (
        draw(dropped_channels(len(x1), max_dropped))
        if isinstance(x1, np.ndarray)
        else []
    )
    to_keep = not DataArray(x1, coords=[("x1", x1)]).isin(dropped)
    dropped_result = result.isel(x1=dropped)
    result = result.where(to_keep)
    if uncertainty and (rel_sigma or abs_sigma):
        error = rel_sigma * result + abs_sigma
        result.attrs["error"] = error
        dropped_error = rel_sigma * dropped_result + abs_sigma
        dropped_result.attrs["error"] = dropped_error
    result.attrs["dropped"] = dropped_result
    result.attrs["datatype"] = (general_type, specific_type)
    result.attrs["provenance"] = MagicMock()
    result.attrs["partial_provenance"] = MagicMock()
    result.attrs["transform"] = transform
    return result


@composite
def datasets(
    draw,
    data_type=(None, {}),
    coordinates=coordinate_transforms(((1.83, 3.9), (-1.75, 2.0), (50.0, 120.0)), 4, 3),
    data=separable_functions(
        smooth_functions((1.83, 3.9)),
        smooth_functions(-1.75, 2.0),
        smooth_functions(50.0, 120.0),
    ),
    rel_sigma=0.02,
    abs_sigma=-1e-3,
    uncertainty=True,
    max_dropped=0.1,
):
    """Returns a Dataset, with appropriate metadata for use in testing
    calculations.

    Parameters
    ----------
    data_type : Tuple[str, Dict[str, str]]
        The data type of the dataset to be generated. If the first element or
        any value in the dictionary is None then it will be drawn from a
        strategy. If the dictionary is empty then its contents will be drawn
        from a strategy.
    coordinates
        A strategy for generating :py:class:`src.converters.CoordinateTransform`
        objects. If absent, any type of transform could be used.
    data
        A strategy to generate functions which calculate the contents of the
        DataArray from the coordinates.
    rel_sigma
        Standard deviation of relative noise applied to the data
    abs_sigma
        Standard deviation of absolute noise applied to the data
    uncertainty
        If ``True``, generate uncertainty metadata using ``rel_sigma`` and
        ``abs_sigma`` (if they are non-zero).
    max_dropped
        The maximum number of channels to drop, as a fraction of the total
        number of channels.

    """

    def compatible(specific):
        return all(
            [
                (general in dt.COMPATIBLE_DATATYPES[specific])
                for general in data_type[1].values()
                if general
            ]
        )

    specific_type = (
        data_type[0] if data_type[0] else draw(specific_datatypes().filter(compatible))
    )
    if data_type[1]:
        general_type = {
            k: (v if v else draw(specific_datatypes(specific_type)))
            for k, v in data_type[1].items()
        }
    else:
        general_type = draw(
            text(), specific_datatypes(specific_type), min_size=1, max_size=5
        )
    transform = draw(coordinates)
    data = {}
    for key, gtype in general_type.items():
        data[key] = draw(
            data_arrays(
                (gtype, specific_type),
                just(transform),
                data,
                rel_sigma,
                abs_sigma,
                uncertainty,
            )
        )
    return Dataset(
        data,
        attrs={"datatype": (specific_type, general_type), "provenance": MagicMock()},
    )
