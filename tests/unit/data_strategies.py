"""Some strategies for generating artificial data on which computations can be
 performed.

"""

import datetime
from unittest.mock import MagicMock

from hypothesis.strategies import booleans
from hypothesis.strategies import composite
from hypothesis.strategies import dates
from hypothesis.strategies import dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import from_regex
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
import numpy as np
from xarray import DataArray

import indica.datatypes as dt
from .strategies import separable_functions
from .strategies import smooth_functions


@composite
def general_datatypes(draw, specific_datatype=None):
    """A strategy to select one of the general data types defined in
    :py:mod:`indica.datatypes`. If ``specific_datatype`` is present then the
    result will be compatible with that.

    """

    if specific_datatype:
        return draw(sampled_from(dt.COMPATIBLE_DATATYPES[specific_datatype]))
    else:
        return draw(sampled_from(sorted(dt.GENERAL_DATATYPES.keys())))


@composite
def specific_datatypes(draw, general_datatype=None):
    """A strategy to select one of the specific data types defined in
    :py:mod:`indica.datatypes`. If ``general_datatype`` is present then the result
    will be compatible with that.

    """

    def valid_datatype(datatype):
        if general_datatype:
            return general_datatype in dt.COMPATIBLE_DATATYPES[datatype]
        return True

    return draw(
        sampled_from(sorted(dt.SPECIFIC_DATATYPES.keys())).filter(valid_datatype)
    )


@composite
def array_datatypes(draw, allow_none=True):
    """Draws a random array datatype, possibly include None values."""
    specific_type = draw(
        one_of(specific_datatypes(), none()) if allow_none else specific_datatypes()
    )
    general_type = draw(
        one_of(general_datatypes(specific_type), none())
        if allow_none
        else general_datatypes(specific_type)
    )
    return general_type, specific_type


@composite
def incompatible_array_types(draw, datatype):
    """Strategy to generate a datatype for a data array that is
    incompatible with the argument. At least one of the general or
    specific datatype will not match.
    """
    errors = draw(lists(integers(0, 1), min_size=1, unique=True))
    specific_type = draw(
        specific_datatypes().filter(lambda x: 0 not in errors or x != datatype[1])
    )
    general_type = draw(
        general_datatypes(specific_type).filter(
            lambda x: 1 not in errors or x != datatype[0]
        )
    )
    return general_type, specific_type


@composite
def dataset_datatypes(draw, min_size=0, max_size=5, allow_none=True):
    """Draws a random dataset datatype, possibly including None values."""
    specific_type = draw(
        one_of(specific_datatypes(), none()) if allow_none else specific_datatypes()
    )
    contents = draw(
        dictionaries(
            from_regex("[a-z0-9]+", fullmatch=True),
            general_datatypes(specific_type),
            min_size=min_size,
            max_size=max_size,
        )
    )
    return specific_type, contents


@composite
def compatible_dataset_types(draw, datatype):
    """Strategy to generate a datatype for a dataset that is compatible
    with the argument. This means the result contains no variables not
    present in ``datatype``, the specific type of all variables is the
    same as that for the dataset as a whole, and all variables have either
    the same or unconstrained general datatype."""
    result_vars = draw(
        lists(sampled_from(sorted(datatype[1].keys())), min_size=1, unique=True).map(
            lambda keys: {k: datatype[1][k] for k in keys}
        )
    )
    return (
        datatype[0],
        {k: None if draw(booleans()) else v for k, v in result_vars.items()},
    )


@composite
def incompatible_dataset_types(draw, datatype):
    """Strategy to generate a datatype for a dataset that is incompatible
    with the argument. This means the result has a different specific
    type than that of ``datatype``, contains one or more variables not
    present in ``datatype``, or the general type of one or more
    variables does not match those in ``datatype``.

    """
    result_vars = draw(
        lists(sampled_from(sorted(datatype[1].keys())), unique=True).map(
            lambda keys: {k: datatype[1][k] for k in keys}
        )
    )
    errors = draw(lists(integers(0, 2), min_size=1, unique=True))
    specific_type = (
        draw(specific_datatypes().filter(lambda d: d != datatype[0]))
        if 0 in errors
        else datatype[0]
    )
    general_types = {k: None if draw(booleans()) else v for k, v in result_vars.items()}
    if 1 in errors:
        change = draw(
            lists(
                booleans(), min_size=len(result_vars), max_size=len(result_vars)
            ).filter(lambda l: any(l))
        )
        for k in [k for k, c in zip(result_vars, change) if c]:
            v = datatype[1][k]
            general_types[k] = draw(
                general_datatypes(specific_type).filter(lambda d: d != v)
            )
    if 2 in errors:
        for key in draw(
            lists(text().filter(lambda t: t not in datatype[1]), min_size=1, max_size=5)
        ):
            general_types[key] = draw(general_datatypes(specific_type))
    return specific_type, general_types


@composite
def adf11_data(
    draw,
    min_num_densities=3,
    max_num_densities=8,
    min_num_temps=5,
    max_num_temps=15,
    max_z=74,
    quantities=["scd", "acd", "plt", "prb"],
):
    """Generates fake ADF11 data.

    Parameters
    ----------
    min_num_densities
        The minimum number of densities to use.
    max_num_densities
        The maximum number of densities to use.
    min_num_temps
        The minimum number of temperatures to use.
    max_num_temps
        The maximum number of temperatures to use.
    max_z
        The maximum atomic number to use.
    quantities
        The types of quantities from which to select.

    """
    nd = draw(integers(min_num_densities, max_num_densities))
    nt = draw(integers(min_num_temps, max_num_temps))
    min_dens = draw(integers(3, 12))
    max_dens = draw(integers(min_dens + 1, 17))
    densities = DataArray(
        np.logspace(min_dens + 6, max_dens + 6, nd), dims="electron_density"
    )
    min_temp = draw(floats(-2.0, 2.5))
    max_temp = draw(floats(min_temp + 1, 6.0))
    temperatures = DataArray(
        np.logspace(min_temp, max_temp, nt), dims="electron_temperature"
    )
    z = draw(integers(1, 74))
    min_z = draw(integers(1, z))
    max_z = draw(integers(min_z, z))
    ion_states = DataArray(np.arange(min_z - 1, max_z, dtype=int), dims="ion_charges")
    func = draw(
        separable_functions(
            smooth_functions((min_z, max_z + 1), max_val=0.1),
            smooth_functions((min_temp, max_temp), max_val=0.1),
            smooth_functions((min_dens, max_dens), max_val=0.1),
        )
    )
    data = 10 ** np.clip(func(ion_states, temperatures, densities) - 6, -99, 99)
    result = DataArray(data, coords=[ion_states, temperatures, densities])
    q = draw(sampled_from(quantities))
    element_name = [value[2] for value in dt.ELEMENTS.values() if value[0] == z][0]
    result.attrs["datatype"] = (dt.ADF11_GENERAL_DATATYPES[q], element_name)
    result.attrs["provenance"] = MagicMock()
    result.attrs["date"] = draw(
        dates(datetime.date(1940, 1, 1), datetime.date(2020, 12, 31))
    )
    result.name = f"{element_name}_{dt.ADF11_GENERAL_DATATYPES[q]}"
    return result
