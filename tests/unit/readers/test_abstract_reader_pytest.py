"""Test methods present on the base class DataReader."""

from copy import deepcopy
import datetime
from numbers import Number
import os
from typing import Any
from typing import Collection
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
import prov.model as prov
from xarray import DataArray

from .available_quantities import AVAILABLE_QUANTITIES
from .selectors import choose_on_plot
from .selectors import DataSelector
from ..abstractio import BaseIO
from ..converters import FluxSurfaceCoordinates
from ..converters import LinesOfSightTransform
from ..converters import MagneticCoordinates
from ..converters import TransectCoordinates
from ..converters import TrivialTransform
from ..datatypes import ArrayType
from ..numpy_typing import ArrayLike
from ..numpy_typing import RevisionLike
from ..session import hash_vals
from ..session import Session
from ..utilities import to_filename

from collections import defaultdict
from contextlib import contextmanager
from copy import copy
import datetime
import os
from tempfile import TemporaryDirectory
from time import sleep
from unittest.mock import MagicMock
from unittest.mock import patch

from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import composite
from hypothesis.strategies import dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import from_regex
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
from hypothesis.strategies import tuples
import numpy as np
import prov.model as prov
from pytest import approx
from pytest import mark
from xarray import DataArray
from xarray import ones_like

from indica.converters import LinesOfSightTransform
from indica.converters import MagneticCoordinates
from indica.converters import TransectCoordinates
from indica.datatypes import SPECIFIC_ELEMENTS
from indica.utilities import coord_array
from .mock_reader import ConcreteReader
from .mock_reader import MockReader
from ..converters.test_lines_of_sight import los_coordinates_and_axes
from ..converters.test_magnetic import magnetic_coordinates_and_axes
from ..converters.test_transect import transect_coordinates_and_axes
from ..data_strategies import array_dictionaries
from ..data_strategies import data_arrays
from ..data_strategies import equilibrium_data
from ..data_strategies import general_datatypes
from ..data_strategies import specific_datatypes
from ..strategies import machine_dimensions
from ..strategies import sane_floats


nsamples = 10
nts = np.random.randint(10, 50, nsamples)
dts = np.random.uniform(0.001, 1., nsamples)
tstarts = np.random.random(nsamples)
tends = tstarts + dts * nts
revisions = np.random.randint(0, 10, nsamples)
lengths = np.random.randint(0, 20, nsamples)

Rmin, Rmax = 0.2, 3.
zmin, zmax = -1, 1

def generate_data(size, low, high):
    return np.random.uniform(low=low, high=high, size=size)


def expected_data(
    diagnostic:str,
    empty=False,
):
    """Strategy to produce a dictionary of DataArrays of the type that
    could be returned by a read operation.
    """

    if empty:
        return {}

    isample = np.random.randint(0, 10)
    tstart = tstarts[isample]
    tend = tends[isample]
    dt = dts[isample]
    revision = revisions[isample]
    length = lengths[isample]

    low = 100.
    high = 5.e3

    diagnostic_coord = diagnostic + "_coord"

    results = {}
    results["revision"] = revision
    results["length"] = length
    results["R"] = np.sort(np.random.random(length) * Rmax + Rmin)
    results["z"] = np.sort(np.random.random(2) * zmax + zmin)
    results["te"] = generate_data()
    time = coord_array(np.arange(tstart, tend+dt, dt), "t")

    data =

    for array in result.values():
        transform = array.attrs["transform"]
        if hasattr(transform, "equilibrium"):
            del transform.equilibrium
        if isinstance(transform, TransectCoordinates):
            x1name = transform.x1_name
            to_fix = [array]
            if "error" in array.attrs:
                to_fix.append(array.attrs["error"])
            if "dropped" in array.attrs:
                to_fix.append(array.attrs["dropped"])
                if "error" in array.attrs:
                    to_fix.append(array.attrs["dropped"].attrs["error"])
            for a in to_fix:
                a.coords["R"] = DataArray(
                    transform.R_vals.y, coords=[(x1name, a.coords[x1name].values)]
                )
                a.coords["z"] = DataArray(
                    transform.z_vals.y, coords=[(x1name, a.coords[x1name].values)]
                )
        if isinstance(transform, MagneticCoordinates):
            to_fix = [array]
            if "error" in array.attrs:
                to_fix.append(array.attrs["error"])
            if "dropped" in array.attrs:
                to_fix.append(array.attrs["dropped"])
                if "error" in array.attrs:
                    to_fix.append(array.attrs["dropped"].attrs["error"])
            for a in to_fix:
                a.coords["z"] = transform.z_los
    return result

def test_thomson_scattering(data_instrument, uid, revision, time_range, max_freq):
    """Test the get_thomson_scattering method correctly combines and processes
    raw data."""
    data, instrument = data_instrument
    for quantity, array in data.items():
        array.name = instrument + "_" + quantity
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_thomson_scattering(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_thomson_scattering(uid, instrument, revision, quantities)
    reader._get_thomson_scattering.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        assert_any_call(
            reader.create_provenance,
            "thomson_scattering",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


